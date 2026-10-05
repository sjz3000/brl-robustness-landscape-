"""
E2 mechanism attribution (brl_e2.py)
=====================================
On top of a trained backbone, perform "robustness-loss attribution" to answer RQ2: is the
robustness damage from quantization driven mainly by (1) weight quantization or (2) activation
quantization? How do rounding schemes (deterministic vs. stochastic noise) matter?

Requires E1-trained checkpoints (brl_rn18_ce.pth / brl_rn18_at.pth).

Three attribution dimensions:
  1. Weight quantization (w-only): only quantize conv/linear weights, activations full-precision
  2. Activation quantization (a-only): only quantize activations (ReLU output), weights full-precision
  3. Rounding scheme (rounding): round-to-nearest vs. stochastic rounding vs. STE pass-through
     -- separates "deterministic precision reduction" from "stochastic noise" effects

Usage:
    # On GPU, once E1 ckpts are ready:
    python brl_e2.py --ckpt ckpt/brl_rn18_ce.pth --bits 32 16 8 6 4 3 2 \
        --pgd-iter 10 --eps 0.03125 --out brl_e2_ce.json
    python brl_e2.py --ckpt ckpt/brl_rn18_at.pth --bits 32 16 8 6 4 3 2 \
        --pgd-iter 10 --eps 0.03125 --out brl_e2_at.json

Output JSON structure:
  {
    "meta": {...},
    "w_only":   [{"bits":8,"clean":..,"robust":..}, ...],   # only weights quantized
    "a_only":   [{"bits":8,"clean":..,"robust":..}, ...],   # only activations quantized
    "w_and_a":  [{"bits":8,"clean":..,"robust":..}, ...],   # both quantized
    "rounding": {"8": {"round":..,"stochastic":..,"ste":..},
                 "4": {...}, "2": {...}}                    # rounding comparison (under weight quantization)
  }

Depends on: brl_quant.py (same dir), brl_scan.py (reused for load/eval), torch, torchvision.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brl_quant as BQ
from brl_scan import build_model, evaluate, load_cifar10


# ---------------- stochastic rounding / STE weight quantization (for rounding attribution) ----------------
def ptq_weight_stochastic(
    model: nn.Module, bits: int, scheme: str = "sym", per_channel: bool = True
):
    """Stochastic rounding: q = floor(x/scale) + Bernoulli(frac). The noise is random, not deterministic rounding."""
    backup = BQ.make_weight_backup(model)
    for m, w in backup.items():
        if bits >= 32:
            continue
        if bits == 16:
            m.weight.data = w.half().float()
            continue
        qmax = float(2 ** (bits - 1) - 1) if scheme == "sym" else float(2 ** bits - 1)
        xr = w.reshape(w.shape[0], -1)
        amax = xr.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        scale = amax / qmax
        qf = xr / scale
        fl = torch.floor(qf)
        frac = qf - fl
        rnd = (torch.rand_like(frac) < frac).float()
        q = fl + rnd
        q = q.clamp(-qmax, qmax)
        dq = q * scale
        m.weight.data = dq.reshape(w.shape)
    return model


def ptq_weight_ste(
    model: nn.Module, bits: int, scheme: str = "sym", per_channel: bool = True
):
    """STE rounding: keep the integer part directly and use STE pass-through on the fractional part
    (forward = quantized, but numerically rounding + pass-through gives a ~0 difference). In
    evaluation mode, only the forward quantized part is used, i.e. round-to-nearest
    (deterministic) — the same as round. To differentiate, this function uses truncation
    (toward zero) to compare "different deterministic mappings"."""
    backup = BQ.make_weight_backup(model)
    for m, w in backup.items():
        if bits >= 32 or bits == 16:
            m.weight.data = w
            continue
        qmax = float(2 ** (bits - 1) - 1) if scheme == "sym" else float(2 ** bits - 1)
        xr = w.reshape(w.shape[0], -1)
        amax = xr.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        scale = amax / qmax
        q = torch.trunc(xr / scale).clamp(-qmax, qmax)  # truncation toward zero (common in STE)
        dq = q * scale
        m.weight.data = dq.reshape(w.shape)
    return model


def main():
    ap = argparse.ArgumentParser(description="BRL E2 mechanism attribution")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--bits", type=int, nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eps", type=float, default=8 / 255)
    ap.add_argument("--pgd-iter", type=int, default=10)
    ap.add_argument("--scheme", default="sym", choices=["sym", "asym"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--round-bits", type=int, nargs="+", default=[8, 4, 2])
    ap.add_argument("--out", default="brl_e2_out.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    bits_list = args.bits if args.bits else BQ.BITWIDTHS

    ds = load_cifar10(args.data_root, args.limit)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    base = build_model(args.model, 10).to(device)
    base.load_state_dict(torch.load(args.ckpt, map_location=device))
    base.eval()
    print(f"[E2] ckpt={args.ckpt} device={device} bits={bits_list}", flush=True)

    # full-precision reference
    clean_fp, robust_fp, _ = evaluate(
        copy.deepcopy(base), loader, device, eps=args.eps, pgd_iters=args.pgd_iter)
    print(f"  FP32: clean={clean_fp:.2f}% robust={robust_fp:.2f}%", flush=True)

    ref = {"fp32_clean": round(clean_fp, 3), "fp32_robust": round(robust_fp, 3)}
    w_only, a_only, w_and_a = [], [], []

    for bits in bits_list:
        # 1) weights only quantized
        m = copy.deepcopy(base)
        if bits == 16:
            m = BQ.set_fp16(m)
        elif bits < 32:
            m, _ = BQ.ptq_quantize_weights(m, bits, scheme=args.scheme,
                                           per_channel=True, inplace=True)
        c, r, _ = evaluate(m, loader, device, eps=args.eps, pgd_iters=args.pgd_iter)
        w_only.append({"bits": bits, "clean": round(c, 3), "robust": round(r, 3)})
        print(f"  [w-only]  bits={bits:>3} clean={c:6.2f}% robust={r:6.2f}%", flush=True)

        # 2) activations only quantized
        m = copy.deepcopy(base)
        restore = BQ.quantize_activation_inplace(m, bits, scheme=args.scheme,
                                                 per_channel=False, modules=(nn.ReLU,))
        c, r, _ = evaluate(m, loader, device, eps=args.eps, pgd_iters=args.pgd_iter)
        restore()
        a_only.append({"bits": bits, "clean": round(c, 3), "robust": round(r, 3)})
        print(f"  [a-only]  bits={bits:>3} clean={c:6.2f}% robust={r:6.2f}%", flush=True)

        # 3) weights and activations both quantized
        m = copy.deepcopy(base)
        m, _ = BQ.ptq_quantize_weights(m, bits, scheme=args.scheme,
                                       per_channel=True, inplace=True)
        restore = BQ.quantize_activation_inplace(m, bits, scheme=args.scheme,
                                                 per_channel=False, modules=(nn.ReLU,))
        c, r, _ = evaluate(m, loader, device, eps=args.eps, pgd_iters=args.pgd_iter)
        w_and_a.append({"bits": bits, "clean": round(c, 3), "robust": round(r, 3)})
        print(f"  [w&a]    bits={bits:>3} clean={c:6.2f}% robust={r:6.2f}%", flush=True)

    # 4) rounding comparison (weights only, key bitwidths)
    rounding = {}
    for bits in args.round_bits:
        if bits >= 16:
            continue
        row = {}
        m = copy.deepcopy(base)
        m, _ = BQ.ptq_quantize_weights(m, bits, scheme=args.scheme,
                                       per_channel=True, inplace=True)
        c, r, _ = evaluate(m, loader, device, eps=args.eps, pgd_iters=args.pgd_iter)
        row["round"] = {"clean": round(c, 3), "robust": round(r, 3)}

        m = copy.deepcopy(base)
        ptq_weight_stochastic(m, bits, scheme=args.scheme)
        c, r, _ = evaluate(m, loader, device, eps=args.eps, pgd_iters=args.pgd_iter)
        row["stochastic"] = {"clean": round(c, 3), "robust": round(r, 3)}

        m = copy.deepcopy(base)
        ptq_weight_ste(m, bits, scheme=args.scheme)
        c, r, _ = evaluate(m, loader, device, eps=args.eps, pgd_iters=args.pgd_iter)
        row["ste_trunc"] = {"clean": round(c, 3), "robust": round(r, 3)}
        rounding[str(bits)] = row
        print(f"  [rounding] bits={bits} round={row['round']} "
              f"stoch={row['stochastic']} ste={row['ste_trunc']}", flush=True)

    out = {
        "meta": {"ckpt": args.ckpt, "model": args.model, "scheme": args.scheme,
                 "eps": args.eps, "pgd_iters": args.pgd_iter, "seed": args.seed},
        "reference_fp32": ref,
        "w_only": w_only, "a_only": a_only, "w_and_a": w_and_a,
        "rounding": rounding,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()

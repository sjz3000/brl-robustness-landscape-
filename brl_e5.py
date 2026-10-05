"""
E5 mixed-precision, robustness-aware bitwidth allocation (brl_e5.py)
================================================
RQ4: under a robustness objective, layers differ in sensitivity to bitwidth compression - which layers must keep high bitwidth and which can be lowered?
E5 proceeds in two parts:
  Part A per-layer robustness-sensitivity scan:
    For each Conv2d/Linear layer of ResNet-18, quantize only that layer's weights to INT3 (others stay full-precision),
    measuring the clean/robust (PGD-20) loss -> a per-layer robustness-loss spectrum, ranked to identify robustness-critical layers.
  Part B robustness-aware mixed-precision allocation:
    Given an average bitwidth budget (default ~4.0 bit), keep the most robustness-sensitive layers (~20% cumulative params) at INT8,
    lowering the rest to INT3 so the parameter-weighted average bitwidth ~= 4.0; compare clean/robust with uniform INT4 (all layers 4.0),
    showing that, at similar average bitwidth, robustness-aware allocation preserves robustness better than uniform allocation.

Depends on: brl_quant.py (same dir), and brl_e3.py's evaluate/pgd (reusing the fixed _norm_pixel_bounds).
Usage (GPU, once the AT backbone is ready):
    python brl_e5.py --ckpt ckpt/brl_rn18_at.pth --pgd-iter 20 --eps 0.03125 --out brl_e5.json
"""
from __future__ import annotations
import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brl_quant as BQ
from brl_scan import build_model, load_cifar10
from brl_e3 import _norm_pixel_bounds, pgd_attack, evaluate_clean_robust

# module types replaced/restored during quantization
MOD = (nn.Conv2d, nn.Conv1d, nn.Linear)


def named_weight_modules(model: nn.Module):
    """Return a (name, module) list containing only weight-bearing conv/linear layers."""
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, MOD) and hasattr(m, "weight")]


def apply_layer_bits(model: nn.Module, layer_bits, backup):
    """Apply quantization per layer according to {name: bits} (in place). bits>=32 keeps full precision. Quantization uses the original weights restored from backup."""
    for name, bits in layer_bits.items():
        m = model
        for part in name.split("."):
            m = getattr(m, part)
        if bits is not None and bits < 32:
            with torch.no_grad():
                m.weight.data = BQ.quantize_tensor(
                    backup[m], bits, scheme="sym", per_channel=True, ch_dim=0)


def main():
    ap = argparse.ArgumentParser(description="BRL E5 mixed-precision robustness-aware bitwidth allocation")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eps", type=float, default=8 / 255)
    ap.add_argument("--pgd-iter", type=int, default=20)
    ap.add_argument("--single-lb", type=int, default=3, help="Part A single-layer quantization bitwidth")
    ap.add_argument("--key-param-frac", type=float, default=0.20,
                    help="Part B cumulative parameter fraction of sensitive layers kept at INT8")
    ap.add_argument("--nonkey-bits", type=int, default=3, help="Part B non-critical-layer bitwidth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="brl_e5.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = load_cifar10(args.data_root, args.limit)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    base = build_model(args.model, 10).to(device)
    base.load_state_dict(torch.load(args.ckpt, map_location=device))
    base.eval()
    wmods = named_weight_modules(base)
    nparam = sum(m.weight.numel() for _, m in wmods)
    total_param = sum(p.numel() for p in base.parameters())
    print(f"[E5] ckpt={args.ckpt} device={device} layers={len(wmods)} "
          f"layer_param={nparam} total_param={total_param}", flush=True)

    # ---- full-precision baseline ----
    c0, r0 = evaluate_clean_robust(copy.deepcopy(base), loader, device,
                                   eps=args.eps, pgd_iters=args.pgd_iter)
    print(f"[E5] FP32: clean={c0:.2f}% robust={r0:.2f}%", flush=True)

    # ---- Part A: per-layer single-layer quantization (INT3) robustness sensitivity ----
    backup = BQ.make_weight_backup(base)
    sens = []
    for name, m in wmods:
        t0 = time.time()
        b = copy.deepcopy(base)
        with torch.no_grad():
            b.load_state_dict(base.state_dict())
            mm = b
            for part in name.split("."):
                mm = getattr(mm, part)
            mm.weight.data = BQ.quantize_tensor(
                backup[m], args.single_lb, scheme="sym", per_channel=True, ch_dim=0)
        c, r = evaluate_clean_robust(b, loader, device,
                                     eps=args.eps, pgd_iters=args.pgd_iter)
        dclean, drob = c0 - c, r0 - r
        sens.append({"layer": name, "param": m.weight.numel(),
                     "clean": round(c, 3), "robust": round(r, 3),
                     "d_clean": round(dclean, 3), "d_robust": round(drob, 3),
                     "time_s": round(time.time() - t0, 2)})
        print(f"  [A] INT3 {name:<20} clean={c:6.2f} robust={r:6.2f} "
              f"dRob={drob:+6.2f} ({time.time()-t0:.0f}s)", flush=True)
        del b

    # sort by robustness loss descending (larger loss = more critical)
    sens.sort(key=lambda x: -x["d_robust"])
    print("\n[E5] Robustness-sensitivity ranking (descending; earlier = more critical, keep high bitwidth):", flush=True)
    for s in sens:
        print(f"  {s['layer']:<24} dRob={s['d_robust']:+6.2f}  param={s['param']}", flush=True)

    # ---- Part B: robustness-aware mixed precision vs. uniform INT4 ----
    # B1: uniform INT4 (all layers at 4 bits)
    b_uniform = copy.deepcopy(base)
    ub = BQ.make_weight_backup(b_uniform)
    for name, m in named_weight_modules(b_uniform):
        with torch.no_grad():
            m.weight.data = BQ.quantize_tensor(ub[m], 4, scheme="sym",
                                               per_channel=True, ch_dim=0)
    cu, ru = evaluate_clean_robust(b_uniform, loader, device,
                                   eps=args.eps, pgd_iters=args.pgd_iter)
    print(f"[B] uniform INT4: clean={cu:.2f}% robust={ru:.2f}%", flush=True)

    # B2: mixed config - keep the most sensitive layers (cumulative ~key_frac) at INT8, lower the rest to nonkey_bits
    cum, key_names, total_key_param = 0.0, [], 0.0
    for s in sens:  # sens is already sorted by d_robust descending
        if cum >= args.key_param_frac:
            break
        key_names.append(s["layer"])
        cum += s["param"] / nparam
        total_key_param += s["param"]
    print(f"[B] critical layers (kept at INT8, cumulative param fraction={cum:.3f}): {key_names}", flush=True)

    layer_bits = {}
    mix_param = 0.0
    for name, m in wmods:
        if name in key_names:
            layer_bits[name] = None  # keep full precision (INT8+ counts as full precision)
            mix_param += 8.0 * m.weight.numel()
        else:
            layer_bits[name] = args.nonkey_bits
            mix_param += float(args.nonkey_bits) * m.weight.numel()
    avg_bits = mix_param / nparam

    b_mix = copy.deepcopy(base)
    mb = BQ.make_weight_backup(b_mix)
    apply_layer_bits(b_mix, layer_bits, mb)
    cm, rm = evaluate_clean_robust(b_mix, loader, device,
                                   eps=args.eps, pgd_iters=args.pgd_iter)
    print(f"[B] robustness-aware mixed (avg={avg_bits:.2f}bits): clean={cm:.2f}% robust={rm:.2f}%", flush=True)

    # ---- output ----
    out = {
        "meta": {"ckpt": args.ckpt, "model": args.model, "pgd_iters": args.pgd_iter,
                 "eps": args.eps, "single_lb": args.single_lb,
                 "key_param_frac": args.key_param_frac, "nonkey_bits": args.nonkey_bits,
                 "seed": args.seed},
        "fp32_reference": {"clean": round(c0, 3), "robust": round(r0, 3)},
        "per_layer_sensitivity": sens,
        "key_layers_int8": [{"layer": n, "param": (lambda s: next(x["param"] for x in sens if x["layer"] == n))(n)} for n in key_names],
        "mixed_precision": {
            "avg_bits": round(avg_bits, 3),
            "layer_bits": {k: (8 if v is None else v) for k, v in layer_bits.items()},
            "uniform_int4": {"clean": round(cu, 3), "robust": round(ru, 3)},
            "mixed_robust": {"clean": round(cm, 3), "robust": round(rm, 3)},
            "robust_gain_vs_uniform": round(rm - ru, 3),
        },
        "landscape_ready": True,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()

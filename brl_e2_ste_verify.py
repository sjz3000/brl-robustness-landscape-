#!/usr/bin/env python3
"""M1 review-gap verification: activation-only "high robustness" is a gradient artifact.

E2 attributes the robustness cliff mainly to weight quantization; the apparent
"high robustness" of isolated activation quantization (a-only, ~85--88%) is argued
to be a PGD-gradient artifact (the piecewise-constant quantizer leaves no gradient
direction). This script closes the verification gap by attacking the SAME a-only
configuration with a straight-through-estimator (STE)-steered PGD, which restores a
valid gradient through the quantized activations, and showing that the inflated
robustness is largely broken:

  mode "std" : deterministic activation quantization, no STE (E2's original attack)
  mode "ste" : STE-steered PGD (gradient treated as identity through quantization)

Run (GPU recommended):
    python brl_e2_ste_verify.py --ckpt ckpt/brl_rn18_at.pth --bits 4 3

Depends on: brl_quant.py, brl_scan.py (same directory).
"""
import argparse
import copy
import json
import numpy as np
import torch
import torch.nn as nn

from brl_quant import quantize_tensor, BITWIDTHS
from brl_scan import build_model, evaluate, load_cifar10


def apply_activation_std(model, bits, scheme="sym"):
    """Deterministic activation quantization, no STE (gradient disabled by quantizer)."""
    return _patch_relu(model, bits, scheme, ste=False)


def apply_activation_ste(model, bits, scheme="sym"):
    """STE activation quantization: forward uses quantized value, gradient passes through."""
    return _patch_relu(model, bits, scheme, ste=True)


def _patch_relu(model, bits, scheme, ste):
    import types

    handles = []
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            orig_fwd = m.forward

            def _new_fwd(_self, inp, _bits=bits, _scheme=scheme, _ste=ste, _of=orig_fwd):
                out = _of(inp)
                q = quantize_tensor(out, _bits, scheme=_scheme, per_channel=False, ch_dim=1)
                if _ste:
                    # straight-through: forward uses q, backward treats gradient as identity
                    return out + (q - out).detach()
                return q

            m.forward = types.MethodType(_new_fwd, m)
            handles.append((m, orig_fwd))

    def _restore():
        for m, f in handles:
            m.forward = f

    return _restore


def main():
    ap = argparse.ArgumentParser(description="M1: activation-only pseudo-robustness under STE-steered PGD")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 3])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eps", type=float, default=8 / 255)
    ap.add_argument("--pgd-iter", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="brl_e2_ste_verify.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_cifar10(args.data_root, args.limit)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    base = build_model(args.model, 10).to(device)
    base.load_state_dict(torch.load(args.ckpt, map_location=device))
    base.eval()
    print(f"[M1-verify] ckpt={args.ckpt} device={device} bits={args.bits} pgd-iter={args.pgd_iter}", flush=True)

    results = {}
    for bits in args.bits:
        # std PGD (original E2 attack)
        m = copy.deepcopy(base)
        restore = apply_activation_std(m, bits)
        c_std, r_std, _ = evaluate(m, loader, device, eps=args.eps, pgd_iters=args.pgd_iter)
        restore()
        # STE-steered PGD
        m = copy.deepcopy(base)
        restore = apply_activation_ste(m, bits)
        c_ste, r_ste, _ = evaluate(m, loader, device, eps=args.eps, pgd_iters=args.pgd_iter)
        restore()
        print(f"  a-only bits={bits}: std_robust={r_std:.2f}%  ste_robust={r_ste:.2f}%  (drop={r_std - r_ste:.2f}pt)", flush=True)
        results[str(bits)] = {
            "std_clean": round(c_std, 3), "std_robust": round(r_std, 3),
            "ste_clean": round(c_ste, 3), "ste_robust": round(r_ste, 3),
        }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[M1-verify] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

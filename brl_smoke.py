"""
P0 BRL smoke main script (brl_smoke.py)
=========================================
One-click scan of the first bitwidth-robustness landscape (BRL) curve: on a given backbone/dataset,
apply PTQ (weight quantization) over a continuous bitwidth spectrum and evaluate clean and PGD-robust accuracy.

Usage (GPU):
    python brl_smoke.py --model resnet18 --dataset cifar10 \
        --bits 32 16 8 6 4 3 2 --pgd-iter 10 --eps 8 --out brl_smoke_out.json

Common:
    # quick smoke test (pipeline check, ~minutes)
    python brl_smoke.py --model resnet18 --dataset cifar10 --limit 500 --pgd-iter 5 \
        --bits 32 8 4 2 --device cuda
    # full scan (landscape)
    python brl_smoke.py --model resnet18 --dataset cifar10 --bits 32 16 8 6 4 3 2

Output JSON:
    {
      "meta": {...},
      "per_bitwidth": [
        {"bits":32,"clean":..,"robust_pgd10":..,"n_test":..,"time_s":..}, ...
      ],
      "landscape_ready": true
    }

Depends on: brl_quant.py (same directory), torch, torchvision, numpy.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brl_quant import BITWIDTHS, ptq_quantize_weights, restore_weights, set_fp16


# ---------------- data ----------------
def load_cifar10(data_root: str, limit: int | None = None):
    tr = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    ds = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=tr)
    if limit:
        ds = torch.utils.data.Subset(ds, np.arange(min(limit, len(ds))))
    return ds


def load_cifar100(data_root: str, limit: int | None = None):
    tr = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    ds = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=tr)
    if limit:
        ds = torch.utils.data.Subset(ds, np.arange(min(limit, len(ds))))
    return ds


def build_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    if name == "resnet18":
        m = torchvision.models.resnet18(num_classes=num_classes, weights=None)
    elif name == "resnet50":
        m = torchvision.models.resnet50(num_classes=num_classes, weights=None)
    elif name == "mobilenetv2":
        m = torchvision.models.mobilenet_v2(num_classes=num_classes, weights=None)
    elif name == "efficientnetb0":
        m = torchvision.models.efficientnet_b0(num_classes=num_classes, weights=None)
    else:
        raise ValueError(f"unknown model {name}")
    m.eval()
    return m


# ---------------- attack ----------------
def pgd_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
               eps: float, step: float, iters: int,
               norm: str = "linf") -> torch.Tensor:
    """White-box PGD. Returns adversarial examples (L_inf bounded)."""
    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(iters):
        out = model(x + delta)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()
        g = delta.grad.data
        delta.data = (delta.data + step * g.sign()).clamp(-eps, eps)
        delta.data = (delta.data + x).clamp(0, 1) - x  # keep within the valid pixel domain
        delta.grad.zero_()
    return x + delta


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, eps=8 / 255, pgd_iters=10,
             step_ratio=2 / 255 / (8 / 255)) -> tuple[float, float, int]:
    """Return (clean_acc%, robust_acc%, n_samples)."""
    model.to(device).eval()
    # type alignment: in the FP16 branch weights are half, inputs must be cast to the same dtype
    wdtype = next(model.parameters()).dtype
    correct_c = 0
    correct_r = 0
    total = 0
    it = iter(loader)
    for images, labels in it:
        images = images.to(device=device, dtype=wdtype)
        labels = labels.to(device)
        # clean
        out = model(images)
        correct_c += (out.argmax(1) == labels).sum().item()
        # adversarial (PGD needs gradients; computed separately, outside no_grad)
        with torch.enable_grad():
            adv = pgd_attack(model, images, labels, eps=eps, step=eps * step_ratio,
                             iters=pgd_iters)
        out_adv = model(adv)
        correct_r += (out_adv.argmax(1) == labels).sum().item()
        total += images.size(0)
    clean = 100.0 * correct_c / total if total else 0.0
    robust = 100.0 * correct_r / total if total else 0.0
    return clean, robust, total


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="BRL smoke: first bitwidth-robustness landscape curve")
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--bits", type=int, nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None, help="number of test samples (smoke test)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eps", type=float, default=8 / 255)
    ap.add_argument("--pgd-iter", type=int, default=10)
    ap.add_argument("--scheme", default="sym", choices=["sym", "asym"])
    ap.add_argument("--per-channel", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="brl_smoke_out.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    bits_list = args.bits if args.bits else BITWIDTHS

    if args.dataset == "cifar10":
        ds = load_cifar10(args.data_root, args.limit)
        n_cls = 10
    else:
        ds = load_cifar100(args.data_root, args.limit)
        n_cls = 100
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # FP32 reference model (random init; the smoke test only validates the pipeline, not accuracy)
    model = build_model(args.model, n_cls, pretrained=False).to(device)
    print(f"[{args.model}@{args.dataset}] device={device} n_cls={n_cls} "
          f"n_test={len(ds)} bits={bits_list}", flush=True)

    results = []
    for bits in bits_list:
        t0 = time.time()
        if bits == 16:
            m = set_fp16(copy_deep(model))
        elif bits >= 32:
            m = copy_deep(model)
        else:
            m, backup = ptq_quantize_weights(copy_deep(model), bits,
                                             scheme=args.scheme,
                                             per_channel=bool(args.per_channel),
                                             inplace=True)
        clean, robust, n = evaluate(m, loader, device, eps=args.eps,
                                    pgd_iters=args.pgd_iter)
        dt = time.time() - t0
        results.append({
            "bits": bits, "clean": round(clean, 3),
            f"robust_pgd{args.pgd_iter}": round(robust, 3),
            "n_test": n, "time_s": round(dt, 2),
        })
        print(f"  bits={bits:>3} clean={clean:6.2f}%  robust_pgd={robust:6.2f}% "
              f"({dt:.1f}s)", flush=True)

    out = {
        "meta": {"model": args.model, "dataset": args.dataset,
                 "scheme": args.scheme, "per_channel": bool(args.per_channel),
                 "eps": args.eps, "pgd_iters": args.pgd_iter,
                 "device": device, "seed": args.seed},
        "per_bitwidth": results,
        "landscape_ready": True,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.out}")


def copy_deep(m: nn.Module) -> nn.Module:
    import copy
    return copy.deepcopy(m).eval()


if __name__ == "__main__":
    main()

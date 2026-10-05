"""
E1 BRL main landscape scan (brl_scan.py)
==========================================
Loads a trained backbone ckpt, applies PTQ (weight quantization) over a continuous bitwidth
spectrum, and evaluates clean accuracy and PGD-robust accuracy to produce the
"bitwidth-robustness landscape" curve. This is the core scientific output of E1.

Usage:
    # Scan CE backbone (standard-trained model)
    python brl_scan.py --ckpt ckpt/brl_rn18_ce.pth --model resnet18 --dataset cifar10 \
        --bits 32 16 8 6 4 3 2 --pgd-iter 10 --eps 0.03125 --out brl_ce_pgd10.json
    # Scan AT backbone (adversarially-trained model)
    python brl_scan.py --ckpt ckpt/brl_rn18_at.pth --model resnet18 --dataset cifar10 \
        --bits 32 16 8 6 4 3 2 --pgd-iter 10 --eps 0.03125 --out brl_at_pgd10.json

Depends on: brl_quant.py (same directory), torch, torchvision.
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
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brl_quant import BITWIDTHS, ptq_quantize_weights, set_fp16


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
    """E7b cross-dataset: CIFAR-100 test set (same normalization as CIFAR-10)."""
    tr = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    ds = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=tr)
    if limit:
        ds = torch.utils.data.Subset(ds, np.arange(min(limit, len(ds))))
    return ds


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "resnet18":
        m = torchvision.models.resnet18(num_classes=num_classes)
    elif name == "resnet50":
        m = torchvision.models.resnet50(num_classes=num_classes)
    elif name == "mobilenetv2":
        m = torchvision.models.mobilenet_v2(num_classes=num_classes)
    elif name == "efficientnetb0":
        m = torchvision.models.efficientnet_b0(num_classes=num_classes)
    else:
        raise ValueError(f"unknown model {name}")
    m.eval()
    return m


def _norm_pixel_bounds(device, dtype=torch.float32):
    # dtype follows input x: avoid type promotion when an fp16 input is clamped to fp32 bounds;
    # otherwise delta becomes fp32, causing a forward input/weight dtype mismatch crash
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device, dtype=dtype).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=device, dtype=dtype).view(3, 1, 1)
    return (-mean / std, (1 - mean) / std)


def pgd_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
               eps: float, step: float, iters: int) -> torch.Tensor:
    low, high = _norm_pixel_bounds(x.device, dtype=x.dtype)
    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(iters):
        out = model(x + delta)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()
        g = delta.grad.data
        delta.data = (delta.data + step * g.sign()).clamp(-eps, eps)
        delta.data = (delta.data + x).clamp(low, high) - x
        delta.grad.zero_()
    return x + delta


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, eps=8 / 255, pgd_iters=10,
             step_ratio=2 / 255 / (8 / 255)) -> tuple[float, float, int]:
    model.to(device).eval()
    wdtype = next(model.parameters()).dtype
    correct_c = correct_r = total = 0
    for images, labels in loader:
        images = images.to(device=device, dtype=wdtype)
        labels = labels.to(device)
        out = model(images)
        correct_c += (out.argmax(1) == labels).sum().item()
        with torch.enable_grad():
            adv = pgd_attack(model, images, labels, eps=eps,
                             step=eps * step_ratio, iters=pgd_iters)
        out_adv = model(adv)
        correct_r += (out_adv.argmax(1) == labels).sum().item()
        total += images.size(0)
    clean = 100.0 * correct_c / total if total else 0.0
    robust = 100.0 * correct_r / total if total else 0.0
    return clean, robust, total


def main():
    ap = argparse.ArgumentParser(description="BRL E1 main landscape scan")
    ap.add_argument("--ckpt", required=True, help="trained backbone weights .pth")
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--bits", type=int, nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eps", type=float, default=8 / 255)
    ap.add_argument("--pgd-iter", type=int, default=10)
    ap.add_argument("--scheme", default="sym", choices=["sym", "asym"])
    ap.add_argument("--per-channel", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="brl_scan_out.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    bits_list = args.bits if args.bits else BITWIDTHS

    if args.dataset == "cifar10":
        ds = load_cifar10(args.data_root, args.limit)
        n_cls = 10
    elif args.dataset == "cifar100":
        ds = load_cifar100(args.data_root, args.limit)
        n_cls = 100
    else:
        raise NotImplementedError(f"unknown dataset {args.dataset}")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Load the trained backbone
    model = build_model(args.model, n_cls).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    print(f"[{args.model}@{args.dataset}] ckpt={args.ckpt} device={device} "
          f"n_test={len(ds)} bits={bits_list}", flush=True)

    results = []
    for bits in bits_list:
        t0 = time.time()
        if bits == 16:
            m = set_fp16(copy.deepcopy(model))
        elif bits >= 32:
            m = copy.deepcopy(model)
        else:
            m, _ = ptq_quantize_weights(copy.deepcopy(model), bits,
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
        "meta": {"model": args.model, "dataset": args.dataset, "ckpt": args.ckpt,
                 "scheme": args.scheme, "per_channel": bool(args.per_channel),
                 "eps": args.eps, "pgd_iters": args.pgd_iter,
                 "device": device, "seed": args.seed},
        "per_bitwidth": results,
        "landscape_ready": True,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()

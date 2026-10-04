"""
方向A · E2 机制归因 (brl_e2.py)
================================
在训练好的骨干上做"鲁棒损失归因", 回答 RQ2: 位宽量化对鲁棒的破坏是
①权重量化主导 还是 ②激活量化主导? 舍入方式(确定性 vs 随机噪声)影响如何?

依赖 E1 训练好的 ckpt (brl_rn18_ce.pth / brl_rn18_at.pth)。

三种归因维度:
  1. 权重量化 (w-only): 仅量化 conv/linear 权重, 激活全精度
  2. 激活量化 (a-only): 仅量化激活(ReLU输出), 权重全精度
  3. 舍入方式 (rounding): 直接取整(round) vs 随机舍入(stochastic) vs STE直通
     —— 分离"确定性精度削减"与"随机噪声"效应

用法:
    # 在 GPU 上, E1 ckpt 就绪后:
    python brl_e2.py --ckpt ckpt/brl_rn18_ce.pth --bits 32 16 8 6 4 3 2 \
        --pgd-iter 10 --eps 0.03125 --out brl_e2_ce.json
    python brl_e2.py --ckpt ckpt/brl_rn18_at.pth --bits 32 16 8 6 4 3 2 \
        --pgd-iter 10 --eps 0.03125 --out brl_e2_at.json

输出 JSON 结构:
  {
    "meta": {...},
    "w_only":   [{"bits":8,"clean":..,"robust":..}, ...],   # 只量化权重
    "a_only":   [{"bits":8,"clean":..,"robust":..}, ...],   # 只量化激活
    "w_and_a":  [{"bits":8,"clean":..,"robust":..}, ...],   # 两者都量化
    "rounding": {"8": {"round":..,"stochastic":..,"ste":..},
                 "4": {...}, "2": {...}}                    # 舍入方式对比(权重量化下)
  }

依赖: brl_quant.py(同目录), brl_scan.py(复用加载/评估), torch, torchvision。
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


# ---------------- 随机舍入 / STE 权重量化(用于舍入归因) ----------------
def ptq_weight_stochastic(
    model: nn.Module, bits: int, scheme: str = "sym", per_channel: bool = True
):
    """随机舍入: q = floor(x/scale) + Bernoulli(frac). 破坏是随机噪声而非确定性取整。"""
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
    """STE 舍入: 整数部分直接保留, 小数部分用 STE 直通(前向=量化, 但数值上取整+直通差分≈0)。
    这里评估模式只取前向量化部分, 即 round-to-nearest(确定性) —— 与 round 相同。
    为区分, 此函数用 truncation(向零截断) 以对比"不同确定性映射"。"""
    backup = BQ.make_weight_backup(model)
    for m, w in backup.items():
        if bits >= 32 or bits == 16:
            m.weight.data = w
            continue
        qmax = float(2 ** (bits - 1) - 1) if scheme == "sym" else float(2 ** bits - 1)
        xr = w.reshape(w.shape[0], -1)
        amax = xr.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        scale = amax / qmax
        q = torch.trunc(xr / scale).clamp(-qmax, qmax)  # 向零截断(STE常用)
        dq = q * scale
        m.weight.data = dq.reshape(w.shape)
    return model


def main():
    ap = argparse.ArgumentParser(description="BRL E2 机制归因")
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

    # 全精度参考
    clean_fp, robust_fp, _ = evaluate(
        copy.deepcopy(base), loader, device, eps=args.eps, pgd_iters=args.pgd_iter)
    print(f"  FP32: clean={clean_fp:.2f}% robust={robust_fp:.2f}%", flush=True)

    ref = {"fp32_clean": round(clean_fp, 3), "fp32_robust": round(robust_fp, 3)}
    w_only, a_only, w_and_a = [], [], []

    for bits in bits_list:
        # 1) 权重仅量化
        m = copy.deepcopy(base)
        if bits == 16:
            m = BQ.set_fp16(m)
        elif bits < 32:
            m, _ = BQ.ptq_quantize_weights(m, bits, scheme=args.scheme,
                                           per_channel=True, inplace=True)
        c, r, _ = evaluate(m, loader, device, eps=args.eps, pgd_iters=args.pgd_iter)
        w_only.append({"bits": bits, "clean": round(c, 3), "robust": round(r, 3)})
        print(f"  [w-only]  bits={bits:>3} clean={c:6.2f}% robust={r:6.2f}%", flush=True)

        # 2) 激活仅量化
        m = copy.deepcopy(base)
        restore = BQ.quantize_activation_inplace(m, bits, scheme=args.scheme,
                                                 per_channel=False, modules=(nn.ReLU,))
        c, r, _ = evaluate(m, loader, device, eps=args.eps, pgd_iters=args.pgd_iter)
        restore()
        a_only.append({"bits": bits, "clean": round(c, 3), "robust": round(r, 3)})
        print(f"  [a-only]  bits={bits:>3} clean={c:6.2f}% robust={r:6.2f}%", flush=True)

        # 3) 权重+激活都量化
        m = copy.deepcopy(base)
        m, _ = BQ.ptq_quantize_weights(m, bits, scheme=args.scheme,
                                       per_channel=True, inplace=True)
        restore = BQ.quantize_activation_inplace(m, bits, scheme=args.scheme,
                                                 per_channel=False, modules=(nn.ReLU,))
        c, r, _ = evaluate(m, loader, device, eps=args.eps, pgd_iters=args.pgd_iter)
        w_and_a.append({"bits": bits, "clean": round(c, 3), "robust": round(r, 3)})
        print(f"  [w&a]    bits={bits:>3} clean={c:6.2f}% robust={r:6.2f}%", flush=True)

    # 4) 舍入方式对比(仅权重量化, 关键位宽)
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

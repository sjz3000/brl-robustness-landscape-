"""
方向A · E3 几何机制 (brl_e3.py)
================================
在训练好的骨干上, 对连续位宽谱的量化模型追踪特征空间几何指标(ID/曲率)随位宽的变化,
检验"位宽压缩 → 特征几何改变(ID↓/曲率变) → 鲁棒损失"的传递链。

复用 lada.manifold_analysis (estimate_intrinsic_dim + compute_curvature_spectrum)。

核心产出 per bitwidth:
  - clean / robust (与 E1 景观对齐)
  - id     (本征维度 TwoNN)
  - curv   (主曲率 mean_curvature)
  - 三者 vs bitwidth 的联合表, 用于 E3 判据:
    "量化是否通过压缩特征几何传导到鲁棒损失"

依赖: brl_quant.py(同目录), lada(同目录), torch, torchvision。

用法(GPU, E1/E2 ckpt 就绪后):
    python brl_e3.py --ckpt ckpt/brl_rn18_ce.pth --bits 32 16 8 6 4 3 2 1 \
        --pgd-iter 20 --eps 0.03125 --feat-layer layer4 --out brl_e3_ce.json
    python brl_e3.py --ckpt ckpt/brl_rn18_at.pth --bits 32 16 8 6 4 3 2 1 \
        --pgd-iter 20 --eps 0.03125 --feat-layer layer4 --out brl_e3_at.json
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
from brl_scan import build_model, load_cifar10

# lada 几何管线 (随包提供, 同目录)
from lada.manifold_analysis import estimate_intrinsic_dim, compute_curvature_spectrum


def _norm_pixel_bounds(device, dtype=torch.float32):
    # 修复 clamp bug: CIFAR 归一化后像素范围非 [0,1], 旧版 clamp(0,1) 会把
    # 扰动投影放大 70-77 倍 → 虚高 robust/崩 clean。改用归一化像素边界。
    # dtype 跟随输入 x: 避免 fp16 输入时被 fp32 边界 clamp 引发的类型提升。
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device, dtype=dtype).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=device, dtype=dtype).view(3, 1, 1)
    low = (0.0 - mean) / std
    high = (1.0 - mean) / std
    return low, high


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


def attach_feature_hook(model: nn.Module, layer_name: str):
    """在指定层后注册 forward hook, 收集特征(不做梯度)。返回 (hook_list, getter)。"""
    feats = {}
    handles = []
    module = model
    for part in layer_name.split("."):
        module = getattr(module, part)
    def _hook(_m, _i, out):
        feats["v"] = out
    handles.append(module.register_forward_hook(_hook))
    def getter():
        return feats.get("v")
    return handles, getter


def extract_features(model: nn.Module, loader, device, layer_name: str,
                     max_n: int = 3000) -> np.ndarray:
    """遍历 loader, 提取指定层输出特征, 返回 (N, D) float32 numpy。"""
    model.eval()
    handles, getter = attach_feature_hook(model, layer_name)
    wdtype = next(model.parameters()).dtype
    allf = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device=device, dtype=wdtype)
            model(images)
            v = getter()
            if v is None:
                continue
            v = v.detach().float().cpu().numpy()
            # 全局平均池化: (B,C,H,W) -> (B,C) 若 4D
            if v.ndim == 4:
                v = v.reshape(v.shape[0], v.shape[1], -1).mean(2)
            allf.append(v)
            if sum(len(x) for x in allf) >= max_n:
                break
    for h in handles:
        h.remove()
    feat = np.concatenate(allf, 0)[:max_n].astype(np.float32)
    return feat


def evaluate_clean_robust(model: nn.Module, loader, device,
                          eps=8 / 255, pgd_iters=10) -> tuple[float, float]:
    model.eval()
    wdtype = next(model.parameters()).dtype
    correct_c = correct_r = total = 0
    it = iter(loader)
    for images, labels in it:
        images = images.to(device=device, dtype=wdtype)
        labels = labels.to(device)
        out = model(images)
        correct_c += (out.argmax(1) == labels).sum().item()
        with torch.enable_grad():
            adv = pgd_attack(model, images, labels, eps=eps,
                             step=eps * (2.0 / 8.0), iters=pgd_iters)
        out_adv = model(adv)
        correct_r += (out_adv.argmax(1) == labels).sum().item()
        total += images.size(0)
    return 100.0 * correct_c / total, 100.0 * correct_r / total


def main():
    ap = argparse.ArgumentParser(description="BRL E3 几何机制")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--bits", type=int, nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eps", type=float, default=8 / 255)
    ap.add_argument("--pgd-iter", type=int, default=10)
    ap.add_argument("--feat-layer", default="layer4")
    ap.add_argument("--max-feat", type=int, default=3000)
    ap.add_argument("--scheme", default="sym", choices=["sym", "asym"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="brl_e3_out.json")
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
    print(f"[E3] ckpt={args.ckpt} device={device} feat_layer={args.feat_layer} "
          f"bits={bits_list}", flush=True)

    # 全精度参考
    c0, r0 = evaluate_clean_robust(copy.deepcopy(base), loader, device,
                                   eps=args.eps, pgd_iters=args.pgd_iter)
    f0 = extract_features(base, loader, device, args.feat_layer, args.max_feat)
    id0 = estimate_intrinsic_dim(f0)
    cv0 = compute_curvature_spectrum(f0)
    print(f"  FP32: clean={c0:.2f}% robust={r0:.2f}% "
          f"id={float(id0):.3f} curv={cv0['mean_curvature']:.4f}", flush=True)

    ref = {"fp32_clean": round(c0, 3), "fp32_robust": round(r0, 3),
           "fp32_id": round(float(id0), 4),
           "fp32_curv": round(float(cv0["mean_curvature"]), 5)}
    rows = []
    for bits in bits_list:
        t0 = time.time()
        m = copy.deepcopy(base)
        if bits == 16:
            m = BQ.set_fp16(m)
        elif 1 < bits < 32:
            m, _ = BQ.ptq_quantize_weights(m, bits, scheme=args.scheme,
                                           per_channel=True, inplace=True)
        # bits==1: 用 brl_quant 的一维二值化量化权重
        elif bits == 1:
            backup = BQ.make_weight_backup(m)
            for mm, w in backup.items():
                mm.weight.data = BQ.quantize_tensor(
                    w, 1, scheme="sym", per_channel=True, ch_dim=0)
        clean, robust = evaluate_clean_robust(m, loader, device,
                                              eps=args.eps, pgd_iters=args.pgd_iter)
        f = extract_features(m, loader, device, args.feat_layer, args.max_feat)
        idf = estimate_intrinsic_dim(f)
        cvf = compute_curvature_spectrum(f)
        mcurv = float(cvf["mean_curvature"])
        rows.append({
            "bits": bits, "clean": round(clean, 3),
            f"robust_pgd{args.pgd_iter}": round(robust, 3),
            "id": round(float(idf), 4), "curv": round(mcurv, 5),
            "time_s": round(time.time() - t0, 2),
        })
        print(f"  bits={bits:>3} clean={clean:6.2f}% robust={robust:6.2f}% "
              f"id={float(idf):6.3f} curv={mcurv:.5f} ({time.time()-t0:.0f}s)",
              flush=True)

    out = {
        "meta": {"ckpt": args.ckpt, "model": args.model, "feat_layer": args.feat_layer,
                 "scheme": args.scheme, "eps": args.eps, "pgd_iters": args.pgd_iter,
                 "max_feat": args.max_feat, "seed": args.seed},
        "reference_fp32": ref,
        "per_bitwidth": rows,
        "landscape_ready": True,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()

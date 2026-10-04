"""
方向A · P1.3 E3 干预实验 (brl_e3_interv.py)
========================================
把 E3 从"相关性"升级为"干预性证据"：对 AT 骨干注入与目标位宽量化误差
等幅(per-layer std 匹配)的随机权重噪声，测量 layer4 特征几何(ID/curv)与
clean/robust，对比"真实量化" vs "同幅随机噪声"。

逻辑：若量化对几何-鲁棒的破坏主要由"可表示分辨率损失导致的权重扰动幅度"
驱动（而非特定离散化阶梯模式），则等幅噪声应复现量化版的 ID↑/curv↓/robust↓ 趋势。
即：量化效应 → 权重扰动幅度 → 几何退化 → 鲁棒损失，提供因果证据。

用法:
    python brl_e3_interv.py --ckpt ckpt/brl_rn18_at.pth --bits 4 3 \
        --pgd-iter 20 --feat-layer layer4 --out brl_e3_interv_at.json
"""
from __future__ import annotations
import argparse, copy, json, os, sys, time
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brl_quant as BQ
from brl_scan import build_model, load_cifar10
from lada.manifold_analysis import estimate_intrinsic_dim, compute_curvature_spectrum
from brl_e3 import _norm_pixel_bounds, pgd_attack, extract_features, evaluate_clean_robust


def inject_matched_noise(model, bits, scheme="sym", per_channel=True, seed=0):
    """对与目标位宽量化参与的所有权重层，注入 per-layer std 匹配该层量化误差的
    随机高斯噪声。返回 (model, stats)。不改变结构，仅扰权重。
    """
    rng = np.random.default_rng(seed)
    backup = BQ.make_weight_backup(model)
    per_layer_std = {}
    with torch.no_grad():
        for m, w in backup.items():
            # 该层量化误差 = |w - quant(w, bits)|
            q = BQ.quantize_tensor(w, bits, scheme=scheme, per_channel=per_channel, ch_dim=0)
            err = (w - q).abs().float()
            sigma = float(err.std().clamp_min(1e-9))
            per_layer_std[m] = sigma
            # 注入独立同分布 Gaussian(0, sigma)
            noise = torch.tensor(rng.standard_normal(w.shape), device=w.device, dtype=w.dtype) * sigma
            m.weight.data = w + noise
    return model, per_layer_std


def main():
    ap = argparse.ArgumentParser(description="P1.3 E3 权重噪声注入干预")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 3])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eps", type=float, default=8 / 255)
    ap.add_argument("--pgd-iter", type=int, default=20)
    ap.add_argument("--feat-layer", default="layer4")
    ap.add_argument("--max-feat", type=int, default=3000)
    ap.add_argument("--scheme", default="sym", choices=["sym", "asym"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="brl_e3_interv.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_cifar10(args.data_root, args.limit)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    base = build_model(args.model, 10).to(device)
    base.load_state_dict(torch.load(args.ckpt, map_location=device))
    base.eval()

    # FP32 参考
    c0, r0 = evaluate_clean_robust(copy.deepcopy(base), loader, device, eps=args.eps, pgd_iters=args.pgd_iter)
    f0 = extract_features(base, loader, device, args.feat_layer, args.max_feat)
    ref = {"fp32_clean": round(c0, 3), "fp32_robust": round(r0, 3),
           "fp32_id": round(float(estimate_intrinsic_dim(f0)), 4),
           "fp32_curv": round(float(compute_curvature_spectrum(f0)["mean_curvature"]), 5)}
    print(f"[interv] ckpt={args.ckpt} bits={args.bits} feat={args.feat_layer} ref={ref}", flush=True)

    rows = []
    for bits in args.bits:
        t0 = time.time()
        # (a) 真实量化
        mq = copy.deepcopy(base)
        mq, _ = BQ.ptq_quantize_weights(mq, bits, scheme=args.scheme, per_channel=True, inplace=True)
        cq, rq = evaluate_clean_robust(mq, loader, device, eps=args.eps, pgd_iters=args.pgd_iter)
        fq = extract_features(mq, loader, device, args.feat_layer, args.max_feat)
        idq = float(estimate_intrinsic_dim(fq)); cvq = float(compute_curvature_spectrum(fq)["mean_curvature"])
        # (b) 同幅随机噪声
        mn, stats = inject_matched_noise(copy.deepcopy(base), bits, scheme=args.scheme, seed=args.seed)
        cn, rn = evaluate_clean_robust(mn, loader, device, eps=args.eps, pgd_iters=args.pgd_iter)
        fn = extract_features(mn, loader, device, args.feat_layer, args.max_feat)
        idn = float(estimate_intrinsic_dim(fn)); cvn = float(compute_curvature_spectrum(fn)["mean_curvature"])
        rows.append({
            "bits": bits,
            "quant": {"clean": round(cq, 3), "robust": round(rq, 3), "id": round(idq, 4), "curv": round(cvq, 5)},
            "noise": {"clean": round(cn, 3), "robust": round(rn, 3), "id": round(idn, 4), "curv": round(cvn, 5),
                      "mean_per_layer_std": round(float(np.mean(list(stats.values()))), 5)},
            "time_s": round(time.time() - t0, 2),
        })
        print(f"  bits={bits}: quant(id={idq:.3f} curv={cvq:.4f} clean={cq:.1f} robust={rq:.1f}) | "
              f"noise(id={idn:.3f} curv={cvn:.4f} clean={cn:.1f} robust={rn:.1f})", flush=True)

    out = {"meta": {"ckpt": args.ckpt, "feat_layer": args.feat_layer, "scheme": args.scheme,
                    "eps": args.eps, "pgd_iters": args.pgd_iter, "seed": args.seed},
           "reference_fp32": ref, "interventions": rows}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()

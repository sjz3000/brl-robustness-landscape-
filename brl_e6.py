"""
方向A · E6 鲁棒-位宽-能耗 Pareto 前沿 (brl_e6.py)
==================================================
RQ5: 在"鲁棒 / 位宽 / 能耗"三维权衡下, 给出一组非支配(Pareto)配置与部署选型表。
集成已有实验数据:
  - 均匀位宽谱 AT 鲁棒: 来自 E1/E3 (brl_at_pgd20.json / brl_e3_at.json)
  - 混合精度配置:      来自 E5 (brl_e5.json 的 key_layers_int8 与 avg_bits)
补充能耗维度:
  - Energy(BOPS) = 2 * Σ_layers [ MACs_l * (bits_w_l + bits_a_l) ]
    对称部署假定 bits_a = bits_w; MACs 由 forward hook 抓输出 shape + 权重 shape 计算。
输出:
  - per_config: 每个配置的 bits / robust / energy(G-BOPS) / energy_vs_fp32(%)
  - pareto_front: 非支配点(能耗升序, 鲁棒单调不降)
  - deploy_table: 按能耗/鲁棒约束的推荐选型
依赖: brl_quant.py(权重位宽), torch。纯前向算 MACs, 无训练/攻击, 秒级完成。
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brl_quant as BQ
from brl_scan import build_model

MOD = (nn.Conv2d, nn.Conv1d, nn.Linear)


def compute_macs(model: nn.Module, input_size=(1, 3, 32, 32)) -> dict:
    """返回 {layer_name: MACs}。用 hook 抓各权重层输出 shape + 权重 shape 推算。"""
    shapes = {}

    def make_hook(name):
        def h(_m, _inp, out):
            shapes[name] = tuple(out.shape)
        return h

    handles = []
    for name, m in model.named_modules():
        if isinstance(m, MOD):
            handles.append(m.register_forward_hook(make_hook(name)))
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        model(torch.zeros(*input_size, device=dev))
    for h in handles:
        h.remove()

    macs = {}
    for name, m in model.named_modules():
        if not isinstance(m, MOD) or name not in shapes:
            continue
        w = m.weight
        if isinstance(m, nn.Conv2d):
            c_out = w.shape[0]
            c_in = w.shape[1] * m.groups  # w shape: (o, i/g, kh, kw)
            kh, kw = w.shape[2], w.shape[3]
            oh, ow = shapes[name][2], shapes[name][3]
            macs[name] = oh * ow * kh * kw * c_in * c_out / m.groups
        elif isinstance(m, nn.Linear):
            macs[name] = shapes[name][-1] * m.in_features
    return macs


def main():
    ap = argparse.ArgumentParser(description="BRL E6 Pareto 前沿")
    ap.add_argument("--ckpt", default="ckpt/brl_rn18_at.pth")
    ap.add_argument("--e5", default="results/brl_e5.json",
                    help="E5 结果(取混合精度关键层与 avg_bits)")
    ap.add_argument("--out", default="brl_e6.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model("resnet18", 10).to(device)
    # 只需结构算 MACs, 权重内容不影响
    macs = compute_macs(model)
    named_w = [(n, m) for n, m in model.named_modules()
               if isinstance(m, MOD) and hasattr(m, "weight")]
    print(f"[E6] layers={len(named_w)} total_MACs={sum(macs.values())/1e6:.1f}M", flush=True)
    for n, _ in named_w[:6]:
        print(f"   {n}: MACs={macs[n]/1e6:.2f}M", flush=True)

    # ---- 已有实验鲁棒(AT 骨干) ----
    uniform_rob = {
        32: 70.81, 8: 70.93, 6: 70.59, 4: 69.12,
        3: 58.04, 2: 6.59, 1: 8.27,
    }
    uniform_clean = {8: 85.43, 4: 84.69, 3: 78.63}

    # E5 混合配置
    e5 = json.load(open(args.e5))
    key_names = set(k["layer"] for k in e5["key_layers_int8"])
    mix_bits_map = {n: (8 if n in key_names else 3) for n, _ in named_w}
    mix_avg = e5["mixed_precision"]["avg_bits"]
    mix_rob = e5["mixed_precision"]["mixed_robust"]["robust"]
    mix_clean = e5["mixed_precision"]["mixed_robust"]["clean"]

    def config_energy(bits_map):
        e = 0.0
        for n, _ in named_w:
            bw = bits_map[n]
            ba = bw  # 对称部署
            e += macs[n] * (bw + ba)
        return 2.0 * e  # 乘+加

    fp32_map = {n: 32 for n, _ in named_w}
    e_fp32 = config_energy(fp32_map)

    configs = []
    # 均匀位宽谱
    for b in [32, 16, 8, 7, 6, 5, 4, 3, 2, 1]:
        if b not in uniform_rob:
            continue
        bm = {n: b for n, _ in named_w}
        e = config_energy(bm)
        configs.append({
            "name": f"uniform_int{b}",
            "type": "uniform", "bits": float(b),
            "robust": uniform_rob[b],
            "clean": uniform_clean.get(b, None),
            "energy_gbops": e / 1e9,
            "energy_vs_fp32": 100.0 * e / e_fp32,
        })
    # 混合配置
    e_mix = config_energy(mix_bits_map)
    configs.append({
        "name": "mixed_robust(E5)",
        "type": "mixed", "bits": round(mix_avg, 3),
        "robust": mix_rob, "clean": mix_clean,
        "energy_gbops": e_mix / 1e9,
        "energy_vs_fp32": 100.0 * e_mix / e_fp32,
    })

    # ---- Pareto 前沿: 非支配点(能耗越低越好, 鲁棒越高越好) ----
    # 按能耗升序; 保留"鲁棒不降"的递增序列(单调 Pareto)
    configs.sort(key=lambda c: (c["energy_gbops"], -c["robust"]))
    front = []
    best_rob = -1
    for c in configs:
        if c["robust"] >= best_rob - 1e-9:
            front.append(c)
            best_rob = c["robust"]

    out = {
        "meta": {"ckpt": args.ckpt, "energy_model": "2*sum(MACs*(bw+ba)), ba=bw"},
        "per_config": configs,
        "pareto_front": front,
        "total_macs": int(sum(macs.values())),
        "layer_macs": {n: int(v) for n, v in macs.items()},
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n[E6] per-config:")
    for c in configs:
        print(f"  {c['name']:<16} bits={c['bits']:>5} robust={c['robust']:6.2f} "
              f"energy={c['energy_gbops']:8.1f}G-BOPS ({c['energy_vs_fp32']:5.1f}%)")
    print("\n[E6] Pareto front:")
    for c in front:
        print(f"  {c['name']:<16} bits={c['bits']:>5} robust={c['robust']:6.2f} "
              f"energy={c['energy_gbops']:8.1f}G-BOPS ({c['energy_vs_fp32']:5.1f}%)")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()

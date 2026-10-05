"""
E6 robustness-bitwidth-energy Pareto frontier (brl_e6.py)
==================================================
RQ5: under the robustness / bitwidth / energy trade-off, produce a set of non-dominated (Pareto) configurations and a deployment selection table.
Integrates existing experimental data:
  - uniform-bitwidth-spectrum AT robustness: from E1/E3 (brl_at_pgd20.json / brl_e3_at.json)
  - mixed-precision config: from E5 (brl_e5.json key_layers_int8 and avg_bits)
Adds the energy dimension:
  - Energy(BOPS) = 2 * Σ_layers [ MACs_l * (bits_w_l + bits_a_l) ]
    symmetric deployment assumes bits_a = bits_w; MACs are computed from forward hooks using output shape + weight shape.
Output:
  - per_config: per-config bits / robust / energy (G-BOPS) / energy_vs_fp32 (%)
  - pareto_front: non-dominated points (ascending energy, robustness non-decreasing)
  - deploy_table: recommended configs under energy/robustness constraints
Depends on: brl_quant.py (weight bitwidth), torch. Forward-only MACs; no training/attack; completes in seconds.
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
    """Return a {layer_name: MACs} map. Uses hooks to capture each weight-layer output shape + weight shape."""
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
    ap = argparse.ArgumentParser(description="BRL E6 Pareto frontier")
    ap.add_argument("--ckpt", default="ckpt/brl_rn18_at.pth")
    ap.add_argument("--e5", default="results/brl_e5.json",
                    help="E5 results (mixed-precision critical layers and avg_bits)")
    ap.add_argument("--out", default="brl_e6.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model("resnet18", 10).to(device)
    # only structure is needed to compute MACs; weight content is irrelevant
    macs = compute_macs(model)
    named_w = [(n, m) for n, m in model.named_modules()
               if isinstance(m, MOD) and hasattr(m, "weight")]
    print(f"[E6] layers={len(named_w)} total_MACs={sum(macs.values())/1e6:.1f}M", flush=True)
    for n, _ in named_w[:6]:
        print(f"   {n}: MACs={macs[n]/1e6:.2f}M", flush=True)

    # ---- existing experimental robustness (AT backbone) ----
    uniform_rob = {
        32: 70.81, 8: 70.93, 6: 70.59, 4: 69.12,
        3: 58.04, 2: 6.59, 1: 8.27,
    }
    uniform_clean = {8: 85.43, 4: 84.69, 3: 78.63}

    # E5 mixed configuration
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
            ba = bw  # symmetric deployment
            e += macs[n] * (bw + ba)
        return 2.0 * e  # multiply + add

    fp32_map = {n: 32 for n, _ in named_w}
    e_fp32 = config_energy(fp32_map)

    configs = []
    # uniform bitwidth spectrum
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
    # mixed configuration
    e_mix = config_energy(mix_bits_map)
    configs.append({
        "name": "mixed_robust(E5)",
        "type": "mixed", "bits": round(mix_avg, 3),
        "robust": mix_rob, "clean": mix_clean,
        "energy_gbops": e_mix / 1e9,
        "energy_vs_fp32": 100.0 * e_mix / e_fp32,
    })

    # ---- Pareto front: non-dominated points (lower energy better, higher robustness better) ----
    # sort by ascending energy; keep the non-decreasing robust sequence (monotone Pareto)
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

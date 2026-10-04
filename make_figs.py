#!/usr/bin/env python3
"""Generate BRL core figures from results JSON (publication-ready, vector PDF)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
figdir = "figs"
os.makedirs(figdir, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})
CW, CH = 3.4, 2.6  # single-column

def load(f):
    with open(os.path.join(R, f)) as fh:
        return json.load(fh)

# ---------- Fig 1: BRL landscape (E1), 4-seed mean +/- std ----------
import statistics as stt
def seed_series(tag):
    agg = {}
    files = [f"brl_{tag}_pgd20.json"] + [f"multiseed_{tag}_s{s}.json" for s in [1,2,3]]
    for f in files:
        d = load(f)
        for x in d["per_bitwidth"]:
            agg.setdefault(x["bits"], []).append(float(x["robust_pgd20"]))
    return agg
ceagg = seed_series("ce"); atagg = seed_series("at")

def meanstd(agg, vb):
    v = agg.get(vb)
    if not v: return None, None
    m = float(np.mean(v))
    s = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return m, s

bits_order = [32,16,8,6,4,3,2,1]
b = [v for v in bits_order if v in ceagg or v in atagg]
x = [np.log2(v) for v in b]
ceR  = [meanstd(ceagg, v)[0] for v in b]
ceRs = [meanstd(ceagg, v)[1] for v in b]
atR  = [meanstd(atagg, v)[0] for v in b]
atRs = [meanstd(atagg, v)[1] for v in b]
# only draw error bars where >1 seed available (INT16/INT1 single-seed -> no bar)
fig, ax = plt.subplots(figsize=(CW, CH), tight_layout=True)
ax.errorbar(x, ceR, yerr=ceRs, fmt="-o", color="#d62728", lw=1.5, ms=3,
            capsize=2.5, elinewidth=0.8, label="CE robust")
ax.errorbar(x, atR, yerr=atRs, fmt="-s", color="#1f77b4", lw=1.5, ms=3,
            capsize=2.5, elinewidth=0.8, label="AT robust")
ax.axvspan(np.log2(3)-0.25, np.log2(3)+0.25, color="gray", alpha=0.15)
ax.annotate("cliff\nINT4$\u2192$INT3", xy=(np.log2(4), atR[b.index(4)]),
            xytext=(np.log2(6.5), 82), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.6))
ax.set_xticks(x); ax.set_xticklabels([str(v) for v in b])
ax.set_xlabel("Bitwidth (FP32$\u2192$INT1)")
ax.set_ylabel("Robust Accuracy (PGD-20, %)")
ax.legend(frameon=False, loc="lower right")
ax.set_ylim(0, 100)
ax.grid(axis="y", ls=":", lw=0.4, alpha=0.5)
fig.savefig(os.path.join(figdir, "fig1_landscape.pdf"))
fig.savefig(os.path.join(figdir, "fig1_landscape.png"), dpi=300)
plt.close(fig)

# ---------- Fig 2: Pareto frontier (E6) ----------
e6 = load("brl_e6.json")
cfg = {c["name"]: c for c in e6["per_config"]}
names = ["uniform_int1","uniform_int3","uniform_int4","uniform_int6","mixed_robust(E5)","uniform_int8"]
en  = [cfg[n]["energy_vs_fp32"] for n in names]
rob = [cfg[n]["robust"] for n in names]
fig, ax = plt.subplots(figsize=(CW, CH), tight_layout=True)
# all points
alln = list(cfg.keys()); ax.plot([cfg[k]["energy_vs_fp32"] for k in alln],
                                 [cfg[k]["robust"] for k in alln], "o", ms=4, color="#555", alpha=0.7)
ax.plot(en, rob, "-", color="#1f77b4", lw=1.5, label="Pareto frontier")
ax.plot(en, rob, "o", color="#1f77b4", ms=4)
for n, dx, dy in [("uniform_int4", 1.2, -4), ("mixed_robust(E5)", -8.5, 2),
                  ("uniform_int8", 2, 3), ("uniform_int1", 1.2, 4)]:
    ax.annotate({"uniform_int4":"INT4","mixed_robust(E5)":"mixed(4.89)","uniform_int8":"INT8","uniform_int1":"INT1"}[n],
                (cfg[n]["energy_vs_fp32"], cfg[n]["robust"]), textcoords="offset points",
                xytext=(dx, dy), fontsize=7)
ax.set_xlabel("Normalized Energy (% of FP32)")
ax.set_ylabel("Robust Accuracy (%)")
ax.set_xlim(0, 30); ax.set_ylim(0, 80)
ax.grid(ls=":", lw=0.4, alpha=0.5)
ax.legend(frameon=False, loc="lower right")
fig.savefig(os.path.join(figdir, "fig2_pareto.pdf"))
fig.savefig(os.path.join(figdir, "fig2_pareto.png"), dpi=300)
plt.close(fig)

# ---------- Fig 3: geometry (E3) ----------
ce3 = load("brl_e3_ce.json"); at3 = load("brl_e3_at.json")
b3 = [x["bits"] for x in ce3["per_bitwidth"]]
x3 = [np.log2(v) for v in b3]
ceCurv=[x["curv"] for x in ce3["per_bitwidth"]]; atCurv=[x["curv"] for x in at3["per_bitwidth"]]
ceID=[x["id"] for x in ce3["per_bitwidth"]];   atID=[x["id"] for x in at3["per_bitwidth"]]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*CW, CH), tight_layout=True)
ax1.plot(x3, ceCurv, "-o", color="#d62728", ms=3, lw=1.4, label="CE")
ax1.plot(x3, atCurv, "-s", color="#1f77b4", ms=3, lw=1.4, label="AT")
ax1.set_xticks(x3); ax1.set_xticklabels([str(v) for v in b3])
ax1.set_xlabel("Bitwidth"); ax1.set_ylabel("Feature curvature")
ax1.axvspan(np.log2(3)-0.25, np.log2(3)+0.25, color="gray", alpha=0.15)
ax1.grid(axis="y", ls=":", lw=0.4, alpha=0.5); ax1.legend(frameon=False, loc="lower left")
ax2.plot(x3, ceID, "-o", color="#d62728", ms=3, lw=1.4, label="CE")
ax2.plot(x3, atID, "-s", color="#1f77b4", ms=3, lw=1.4, label="AT")
ax2.set_xticks(x3); ax2.set_xticklabels([str(v) for v in b3])
ax2.set_xlabel("Bitwidth"); ax2.set_ylabel("Intrinsic dimension (ID)")
ax2.axvspan(np.log2(3)-0.25, np.log2(3)+0.25, color="gray", alpha=0.15)
ax2.grid(axis="y", ls=":", lw=0.4, alpha=0.5); ax2.legend(frameon=False, loc="upper left")
fig.savefig(os.path.join(figdir, "fig3_geometry.pdf"))
fig.savefig(os.path.join(figdir, "fig3_geometry.png"), dpi=300)
plt.close(fig)

print("figs:", os.listdir(figdir))

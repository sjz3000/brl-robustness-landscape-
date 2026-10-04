#!/usr/bin/env python3
"""Reproduce the E1 statistical report in the paper (Sec. E1, "Statistical significance").

Reads the four-seed robust-accuracy JSONs produced by brl_scan.py and computes,
for each bitwidth, the mean +/- std and the 95% confidence interval over the
four training seeds; then reports the cliff (INT4->INT3) paired t-test with
Cohen's d_z and the AT-vs-CE independent (Welch) t-test for the headline
bitwidths, matching the numbers quoted in the manuscript.

Usage:
    python compute_stats.py <ce_seed0.json> <at_seed0.json> [--multiseed <dir>]

The multiseed dir (default: results/) is expected to contain
multiseed_{ce,at}_s1/s2/s3.json alongside the seed-0 files.

Outputs a concise table identical in spirit to the paper's E1 statistics.
"""

import argparse, json, math, os
from statistics import mean, stdev
try:
    import scipy.stats as st
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

T_CRIT_95_DF3 = 3.182  # t_{0.025,3} for a two-sided 95% CI with n=4


def load_tag(tag, seed0_path, mdir):
    data = {}
    d0 = json.load(open(seed0_path))
    for p in d0["per_bitwidth"]:
        data[p["bits"]] = [round(p["robust_pgd20"], 2)]
    for i in (1, 2, 3):
        fp = os.path.join(mdir, f"multiseed_{tag}_s{i}.json")
        if not os.path.exists(fp):
            continue
        d = json.load(open(fp))
        for p in d["per_bitwidth"]:
            if p["bits"] in data:
                data[p["bits"]].append(round(p["robust_pgd20"], 2))
    return data


def ci_95(vals):
    m = mean(vals)
    s = stdev(vals)
    se = s / math.sqrt(len(vals))
    return m, s, (m - T_CRIT_95_DF3 * se, m + T_CRIT_95_DF3 * se)


def cohen_dz(x, y):
    diff = [a - b for a, b in zip(x, y)]
    dm = mean(diff)
    ds = stdev(diff)
    return abs(dm) / ds if ds > 0 else float("inf")


def paired_p(x, y):
    diff = [a - b for a, b in zip(x, y)]
    dm = mean(diff)
    sd = stdev(diff)
    n = len(diff)
    t = dm / (sd / math.sqrt(n))
    return abs(t), (2.0 * st.t.cdf(-abs(t), df=n - 1) if HAS_SCIPY else None)


def welch_t(a, b):
    na, nb = len(a), len(b)
    sa, sb = stdev(a), stdev(b)
    t = (mean(a) - mean(b)) / math.sqrt(sa**2 / na + sb**2 / nb)
    df = (sa**2 / na + sb**2 / nb) ** 2 / (
        (sa**2 / na) ** 2 / (na - 1) + (sb**2 / nb) ** 2 / (nb - 1))
    return t, (2.0 * st.t.cdf(-abs(t), df=df) if HAS_SCIPY else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ce_seed0")
    ap.add_argument("at_seed0")
    ap.add_argument("--multiseed", default="results")
    a = ap.parse_args()

    ce = load_tag("ce", a.ce_seed0, a.multiseed)
    at = load_tag("at", a.at_seed0, a.multiseed)

    print("scipy available:", HAS_SCIPY)
    print("\n=== Per-bitwidth 4-seed mean +/- std and 95% CI ===")
    for tag, data in (("AT", at), ("CE", ce)):
        for b in (32, 4, 3, 2):
            if b not in data or len(data[b]) < 2:
                continue
            m, s, (lo, hi) = ci_95(data[b])
            print(f"{tag} INT{b}: mean={m:.2f} std={s:.2f} "
                  f"95%CI=[{lo:.2f},{hi:.2f}] seeds={len(data[b])}")

    print("\n=== Cliff INT4->INT3 (paired t over seeds) ===")
    for tag, data in (("AT", at), ("CE", ce)):
        if 4 in data and 3 in data and len(data[4]) == len(data[3]):
            dz = cohen_dz(data[4], data[3])
            t, p = paired_p(data[4], data[3])
            drop = mean(data[4]) - mean(data[3])
            pstr = f"p={p:.4f}" if p is not None else "(no scipy)"
            print(f"{tag}: drop={drop:.2f}pt  d_z={dz:.2f}  t={t:.1f}  {pstr}")

    print("\n=== AT vs CE (independent Welch t) ===")
    for b in (32, 4, 3):
        if b in at and b in ce:
            t, p = welch_t(at[b], ce[b])
            diff = mean(at[b]) - mean(ce[b])
            pstr = f"p={p:.4f}" if p is not None else "(no scipy)"
            print(f"INT{b}: AT-CE={diff:.2f}pt  t={t:.1f}  {pstr}")


if __name__ == "__main__":
    main()

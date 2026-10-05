# BRL — Bitwidth–Robustness Landscape (Reproducible Code)

Reproduces all experiments of the paper *"Where Does Quantization Break Adversarial
Robustness? A Bitwidth–Robustness Landscape with Edge-Friendly Re-training"*
(Neurocomputing submission).

**Core findings (three laws):**
1. The bitwidth–robustness curve follows a *plateau–cliff–collapse* pattern,
   with the cliff at **INT4→INT3**;
2. **AT (adversarial training)** vastly outperforms **CE** across all bitwidths,
   and retains robustness better at low bitwidths;
3. **INT2/INT1 collapse** (under direct PTQ and standard re-training F1–F3),
   recoverable only via bitwidth-specialized Robust-QAT.

---

## 1. Environment & Installation

```bash
conda create -n brl python=3.10 -y && conda activate brl
pip install -r requirements.txt
```

> The paper experiments ran on a single **RTX 3090 (24 GB)**, PyTorch 2.1.0+cu121.
> Data (CIFAR-10/100) are downloaded automatically by torchvision into `./data`
> (use `--data-root` to change the path).

---

## 2. Code Structure (Reproduction)

| File | Description |
|------|-------------|
| `brl_quant.py` | Core quantization library: arbitrary-bitwidth symmetric quantization, STE pseudo-quantization, PTQ (weight/activation), FP16, weight backup/restore |
| `brl_train.py` | Backbone training: CE (200 ep) / AT (80 ep, PGD-20), ResNet-18 / MobileNetV2, multi-seed support |
| `brl_scan.py` | Bitwidth-spectrum scan: load ckpt → per-bitwidth PTQ → clean/PGD robust → `results/*.json` |
| `compute_stats.py` | **E1 statistical report**: 4-seed mean±std, 95% CI, cliff paired t + Cohen's d, AT vs CE Welch t |
| `brl_e2.py` | E2 mechanism attribution: weight/activation quantization, rounding (round/stochastic/STE-trunc) |
| `brl_e3.py` | E3 geometric mechanism: layer-4 manifold intrinsic dimension (ID) and principal curvature (curv) |
| `brl_e3_interv.py` | **E3b** causal intervention: inject Gaussian noise matched to quantization error, compare ID/curv/robust |
| `brl_e4.py` | **E4** Robust-QAT three paradigms: F1 (AT→PTQ) / F2 (AT→low-bit fine-tune) / F3 (joint) |
| `brl_binary_qat.py` | **E4b** bitwidth-specialized Robust-QAT (RA-BNN style, INT2/INT1 binary baseline) |
| `brl_e5.py` | **E5** mixed precision: per-layer INT3 sensitivity, robustness-aware bitwidth allocation |
| `brl_e6.py` | **E6** robustness–bitwidth–energy Pareto frontier (MACs×bitwidth energy proxy) |
| `brl_bfa.py` | **E7/fault** bit-flip fault injection (weight flips of 10/100 bits) robustness |
| `brl_autoattack_eval.py` | **tab:aa** AutoAttack strong evaluator (APGD-CE+DLR+SQUARE), n=10000 |
| `brl_onnx_bench.py` | ONNX latency proxy benchmark (energy/latency helper) |
| `brl_train_vit.py` / `brl_scan_vit.py` | **E7c** DeiT-Tiny (attention) cross-architecture training and bitwidth spectrum |
| `lada/manifold_analysis.py` | E3/E3b geometry computation (TwoNN intrinsic dimension, principal curvature) |
| `make_figs.py` | Script to generate the three paper figures (landscape/geometry/Pareto) |
| `smoke_e2e.py` / `brl_smoke.py` | End-to-end smoke self-test (synthetic data, quick pipeline check) |

---

## 3. Quick Smoke Test (Verify the Pipeline)

```bash
# End-to-end self-test on synthetic data (no full dataset needed, ~1 min)
python smoke_e2e.py

# 7-bitwidth scan under random weights (verify brl_quant + scan chain)
python brl_scan.py --ckpt <any.pth> --model resnet18 --dataset cifar10 \
    --bits 32 16 8 6 4 3 2 1 --device cuda
```

---

## 4. Full Reproduction (Paper E1→E7 Order)

All commands produce result JSONs under `results/` (matching paper tables `tab:*`).

### 4.1 E1 Backbone Training + Main Landscape (Table tab:e1)
```bash
# CE 200 epochs
python brl_train.py --mode ce --model resnet18 --dataset cifar10 --epochs 200 \
    --out ckpt/brl_rn18_ce.pth --log train_ce.log
# AT 80 epochs (PGD-20)
python brl_train.py --mode at --model resnet18 --dataset cifar10 --epochs 80 \
    --out ckpt/brl_rn18_at.pth --log train_at.log
# 4-seed mean±std (train seed 1/2/3 separately, combine for statistics)
for s in 0 1 2 3; do
  python brl_train.py --mode at --model resnet18 --dataset cifar10 --epochs 80 \
      --seed $s --out ckpt/brl_rn18_at_s$s.pth --log train_at_s$s.log
done

# Bitwidth-spectrum scan (CE and AT backbones)
python brl_scan.py --ckpt ckpt/brl_rn18_ce.pth --model resnet18 --dataset cifar10 \
    --bits 32 16 8 6 4 3 2 1 --pgd-iter 20 --out results/brl_ce_pgd20.json
python brl_scan.py --ckpt ckpt/brl_rn18_at.pth --model resnet18 --dataset cifar10 \
    --bits 32 16 8 6 4 3 2 1 --pgd-iter 20 --out results/brl_at_pgd20.json
```
> Multi-seed mean±std: for each bitwidth, compute mean±std of clean/robust over
> s∈{0,1,2,3} (paper Table tab:e1).
> Reproduce the paper's E1 statistical report (95% CI, cliff paired t-test,
> AT vs CE Welch t, effect size):
```bash
python compute_stats.py results/brl_ce_pgd20.json results/brl_at_pgd20.json --multiseed results
```

### 4.2 AutoAttack Cross-Validation (Table tab:aa)
```bash
python brl_autoattack_eval.py --ckpt ckpt/brl_rn18_at.pth --model resnet18 \
    --dataset cifar10 --bits 32 4 3 2 --norm Linf --eps 0.03125 --limit 10000 \
    --out results/brl_at_aa_n10000.json
python brl_autoattack_eval.py --ckpt ckpt/brl_rn18_ce.pth --model resnet18 \
    --dataset cifar10 --bits 32 4 3 2 --norm Linf --eps 0.03125 --limit 10000 \
    --out results/brl_ce_aa_n10000.json
```

### 4.3 E2 Mechanism Attribution (Tables tab:e2w / tab:e2r)
```bash
python brl_e2.py --ckpt ckpt/brl_rn18_at.pth --bits 8 6 4 3 2 --pgd-iter 20 \
    --out results/brl_e2_at.json
python brl_e2.py --ckpt ckpt/brl_rn18_ce.pth --bits 8 6 4 3 2 --pgd-iter 20 \
    --out results/brl_e2_ce.json
```

### 4.4 E3 Geometric Mechanism + E3b Causal Intervention (Tables tab:e3 / tab:e3i)
```bash
# Geometry metrics (ID/curv) vs bitwidth
python brl_e3.py --ckpt ckpt/brl_rn18_at.pth --bits 32 8 6 4 3 2 --feat-layer layer4 \
    --out results/brl_e3_at.json
# Causal intervention: inject noise matched to quantization error
python brl_e3_interv.py --ckpt ckpt/brl_rn18_at.pth --bits 4 3 --feat-layer layer4 \
    --out results/brl_e3_interv_at.json
```

### 4.5 E4 Robust-QAT Three Paradigms (Table tab:e4)
```bash
python brl_e4.py --ckpt ckpt/brl_rn18_at.pth --bits 4 3 2 --at-iters 8 \
    --out results/brl_e4.json
```

### 4.6 E4b Bitwidth-Specialized Robust-QAT / Binary Baseline (Table tab:qat)
```bash
# Bitwidth-specialized QAT at INT2
python brl_binary_qat.py --mode qat --bits 2 --out ckpt/brl_rn18_qat_2.pth
# Bitwidth-specialized QAT at INT1
python brl_binary_qat.py --mode qat --bits 1 --out ckpt/brl_rn18_qat_1.pth
# Scan evaluation
brl_scan.py --ckpt ckpt/brl_rn18_qat_2.pth --model resnet18 --dataset cifar10 \
    --bits 32 8 4 3 2 1 --out results/qat_2_pgd20.json
brl_scan.py --ckpt ckpt/brl_rn18_qat_1.pth --model resnet18 --dataset cifar10 \
    --bits 32 8 4 3 2 1 --out results/qat_1_pgd20.json
```

### 4.7 E5 Mixed Precision (Table tab:e5)
```bash
python brl_e5.py --ckpt ckpt/brl_rn18_at.pth --pgd-iter 20 --out results/brl_e5.json
```

### 4.8 E6 Energy Pareto Frontier (Table tab:e6)
```bash
python brl_e6.py --ckpt ckpt/brl_rn18_at.pth --out results/brl_e6.json
```

### 4.9 E7 Generalization
#### E7a Cross-Architecture — MobileNetV2 (Table tab:e7a)
```bash
python brl_train.py --mode at --model mobilenetv2 --dataset cifar10 --epochs 80 \
    --out ckpt/brl_mv2_at.pth
python brl_scan.py --ckpt ckpt/brl_mv2_at.pth --model mobilenetv2 --dataset cifar10 \
    --bits 32 16 8 6 4 3 2 1 --out results/mv2_at_pgd20.json
```

#### E7b Cross-Dataset — CIFAR-100 (Table tab:e7b)
```bash
python brl_train.py --mode at --model resnet18 --dataset cifar100 --epochs 80 \
    --out ckpt/brl_rn18_c100_at.pth
python brl_scan.py --ckpt ckpt/brl_rn18_c100_at.pth --model resnet18 --dataset cifar100 \
    --bits 32 16 8 6 4 3 2 1 --out results/c100_at_pgd20.json
```

#### E7c Cross-Architecture — DeiT-Tiny (attention, Table tab:e7c, PGD-10)
```bash
python brl_train_vit.py --mode at --model deit_tiny --epochs 80 --lr 0.05 \
    --out ckpt/brl_deit_at.pth
python brl_scan_vit.py --ckpt ckpt/brl_deit_at.pth --model deit_tiny --dataset cifar10 \
    --bits 32 16 8 6 4 3 2 1 --out results/deit_at_pgd20.json
```
> E7c uses PGD-10 (not PGD-20) because position-encoding sensitivity makes PGD-20
> too costly at this resolution; the paper compares relative landscapes only and
> does not compare absolute values across attack budgets.

### 4.10 Fault Injection (Table tab:bfa)
```bash
python brl_bfa.py --ckpt ckpt/brl_rn18_at.pth --out results/bfa_at.json
python brl_bfa.py --ckpt ckpt/brl_rn18_ce.pth --out results/bfa_ce.json
```

### 4.11 ε-Sweep (Table tab:eps, validate theoretical-model predictions)
```bash
for eps in 4/255 8/255 16/255; do
  python brl_scan.py --ckpt ckpt/brl_rn18_at.pth --model resnet18 --dataset cifar10 \
      --bits 8 6 5 4 3 --pgd-iter 20 --eps $eps --out results/eps_${eps//\//_}.json
done
```

---

## 5. Paper Table ↔ Command Quick Map

| Paper table | Experiment | Primary command |
|-------------|-----------|-----------------|
| `tab:e1` | Main landscape (4-seed) | `brl_train`×4 + `brl_scan` + mean±std |
| `tab:aa` | AutoAttack | `brl_autoattack_eval --limit 10000` |
| `tab:e2w/e2r` | Weight/activation mechanism, rounding | `brl_e2` |
| `tab:e3` / `tab:e3i` | Geometry ID/curv, causal intervention | `brl_e3` / `brl_e3_interv` |
| `tab:e4` | Robust-QAT three paradigms | `brl_e4` |
| `tab:qat` | Bitwidth-specialized QAT | `brl_binary_qat` + `brl_scan` |
| `tab:e5` | Mixed-precision sensitivity | `brl_e5` |
| `tab:e6` | Energy Pareto | `brl_e6` |
| `tab:e7a` | MV2 cross-architecture | `brl_train/scan --model mobilenetv2` |
| `tab:e7b` | CIFAR-100 cross-dataset | `brl_train/scan --dataset cifar100` |
| `tab:e7c` | DeiT-Tiny attention | `brl_train_vit/scan_vit --model deit_tiny` |
| `tab:bfa` | Bit-flip fault | `brl_bfa` |
| `tab:eps` | ε-sweep | `brl_scan --eps {4,8,16}/255` |
| `fig:landscape/geometry/pareto` | Paper figures | `make_figs.py` |

---

## 6. Output & Verification

Each script writes `.json` under `results/`. All numbers in the paper tables come
directly from these JSONs with no manual modification — reproducibility means
"re-run the command → obtain the JSON matching the paper → fill the table."

Several counter-intuitive findings are explicitly and honestly flagged (not
experimental error):
- *The "high robustness" of isolated activation quantization is a PGD-gradient
  failure artifact*, not a real robustness gain (E2);
- *INT2 collapse is a real phenomenon*, cross-validated under multiple attack
  budgets (PGD-8/20, AutoAttack, L2).

---

## 7. Reproduction Notes

- **Compute**: A full reproduction of all experiments (including 4-seed training,
  full bitwidth spectra, QAT, and generalization) requires **several GPU-days**
  (RTX 3090). To verify the conclusions alone, run the smoke test first, then the
  E1 single-seed landscape and E2 attribution.
- **Randomness**: Training uses `--seed` for reproducibility; evaluation
  (scan/attribution) defaults to seed=0 for repeatability.
- **DeiT-Tiny**: `timm.create_model('deit_tiny_patch16_224', img_size=32, patch_size=4)`
  requires downloading the pretrained config online (`pretrained=False`, structure only).
- **lada geometry pipeline**: `lada/manifold_analysis.py` is bundled; scripts
  automatically prefer the `lada/` directory located next to them.

---

## 8. License & Citation

The code is provided for academic reproduction only. For citation, see the paper's
references; for questions, contact the corresponding author
(the anonymous review version omits author information; the final version appears
in the author footnote).

# BRL — Bitwidth–Robustness Landscape 论文可复现代码

复现论文 *"Where Does Quantization Break Adversarial Robustness? A Bitwidth–Robustness
Landscape with Edge-Friendly Re-training"*（Neurocomputing 投稿版）的全部实验。

**核心主张**（三大定律）：
1. 位宽鲁棒性呈"平缓–悬崖–崩塌"三段式，拐点在 **INT4→INT3**；
2. **AT（对抗训练）全程大幅优于 CE**，且低位宽时保鲁棒更好；
3. **INT2/INT1 崩塌**（在直接 PTQ 与常规重训练 F1–F3 下），仅位宽专用 Robust-QAT 可恢复。

---

## 1. 环境与安装

```bash
conda create -n brl python=3.10 -y && conda activate brl
pip install -r requirements.txt
```

> 论文原始实验运行于单张 **RTX 3090（24GB）**、PyTorch 2.1.0+cu121。数据（CIFAR-10/100）
> 由脚本经 torchvision 自动下载到 `./data`（可用 `--data-root` 指定其他路径）。

---

## 2. 代码结构（复现用）

| 文件 | 说明 |
|------|------|
| `brl_quant.py` | 核心量化库：任意位宽对称量化、STE 伪量化、PTQ（权重/激活）、FP16、权重备份恢复 |
| `brl_train.py` | 骨干训练：CE（200ep）/ AT（80ep，PGD-20），ResNet-18 / MobileNetV2，可配多 seed |
| `brl_scan.py` | 位宽谱扫描：加载 ckpt → 逐位宽 PTQ → clean/PGD robust → `results/*.json` |
| `compute_stats.py` | **E1 统计报告复现**：4-seed mean±std、95% CI、悬崖配对 t + Cohen's d、AT vs CE Welch t |
| `brl_e2.py` | E2 机制归因：权重/激活量化、舍入方式（round/stochastic/STE-trunc） |
| `brl_e3.py` | E3 几何机制：layer-4 流形的 intrinsic dimension (ID) 与主曲率 (curv) |
| `brl_e3_interv.py` | **E3b** 因果干预：权重注入匹配量化误差的高斯噪声，对比 ID/curv/robust |
| `brl_e4.py` | **E4** Robust-QAT 三范式：F1(AT→PTQ) / F2(AT→低比特微调) / F3(联合) |
| `brl_binary_qat.py` | **E4b** 位宽专用 Robust-QAT（RA-BNN 风格，INT2/INT1 二值基线） |
| `brl_e5.py` | **E5** 混合精度：逐层 INT3 敏感性、robustness-aware 位宽分配 |
| `brl_e6.py` | **E6** 鲁棒–位宽–能量 Pareto frontier（MACs×bitwidth 能量代理） |
| `brl_bfa.py` | **E7/fault** bit-flip 故障注入（权重翻转 10/100 位）鲁棒性 |
| `brl_autoattack_eval.py` | **tab:aa** AutoAttack 强评估器（APGD-CE+DLR+SQUARE），n=10000 |
| `brl_onnx_bench.py` | ONNX 延迟代理基准（energy/latency 辅助） |
| `brl_train_vit.py` / `brl_scan_vit.py` | **E7c** DeiT-Tiny（attention）跨架构训练与位宽谱 |
| `lada/manifold_analysis.py` | E3/E3b 几何量计算（TwoNN intrinsic dimension、主曲率） |
| `make_figs.py` | 论文三图（景观/几何/Pareto）生成脚本 |
| `smoke_e2e.py` / `brl_smoke.py` | 端到端冒烟自检（合成数据，快速验证管线） |

---

## 3. 快速冒烟（验证管线正常）

```bash
# 合成数据端到端自检（不依赖完整数据集，约 1 分钟）
python smoke_e2e.py

# 随机权重下 7 位宽谱扫描（验证 brl_quant+scan 链路）
python brl_scan.py --ckpt <任意.pth> --model resnet18 --dataset cifar10 \
    --bits 32 16 8 6 4 3 2 1 --device cuda
```

---

## 4. 完整复现流程（按论文 E1→E7 顺序）

所有命令生成结果 JSON，输出到 `results/`（与论文表 `tab:*` 对应）。

### 4.1 E1 骨干训练 + 主景观（表 tab:e1）
```bash
# CE 200 轮
python brl_train.py --mode ce --model resnet18 --dataset cifar10 --epochs 200 \
    --out ckpt/brl_rn18_ce.pth --log train_ce.log
# AT 80 轮（PGD-20）
python brl_train.py --mode at --model resnet18 --dataset cifar10 --epochs 80 \
    --out ckpt/brl_rn18_at.pth --log train_at.log
# 4 种子 mean±std（seed 1/2/3 另训，组合计算）
for s in 0 1 2 3; do
  python brl_train.py --mode at --model resnet18 --dataset cifar10 --epochs 80 \
      --seed $s --out ckpt/brl_rn18_at_s$s.pth --log train_at_s$s.log
done

# 位宽谱扫描（CE 与 AT 骨干）
python brl_scan.py --ckpt ckpt/brl_rn18_ce.pth --model resnet18 --dataset cifar10 \
    --bits 32 16 8 6 4 3 2 1 --pgd-iter 20 --out results/brl_ce_pgd20.json
python brl_scan.py --ckpt ckpt/brl_rn18_at.pth --model resnet18 --dataset cifar10 \
    --bits 32 16 8 6 4 3 2 1 --pgd-iter 20 --out results/brl_at_pgd20.json
```
> 多 seed 均值±标准差：对 s∈{0,1,2,3} 的每个位宽 clean/robust 求 mean±std（论文表 tab:e1）。
> 复现论文 E1 的统计报告（95% CI、悬崖配对 t 检验、AT vs CE Welch t、效应量）：
```bash
python compute_stats.py results/brl_ce_pgd20.json results/brl_at_pgd20.json --multiseed results
```

### 4.2 AutoAttack 交叉验证（表 tab:aa）
```bash
python brl_autoattack_eval.py --ckpt ckpt/brl_rn18_at.pth --model resnet18 \
    --dataset cifar10 --bits 32 4 3 2 --norm Linf --eps 0.03125 --limit 10000 \
    --out results/brl_at_aa_n10000.json
python brl_autoattack_eval.py --ckpt ckpt/brl_rn18_ce.pth --model resnet18 \
    --dataset cifar10 --bits 32 4 3 2 --norm Linf --eps 0.03125 --limit 10000 \
    --out results/brl_ce_aa_n10000.json
```

### 4.3 E2 机制归因（表 tab:e2w / tab:e2r）
```bash
python brl_e2.py --ckpt ckpt/brl_rn18_at.pth --bits 8 6 4 3 2 --pgd-iter 20 \
    --out results/brl_e2_at.json
python brl_e2.py --ckpt ckpt/brl_rn18_ce.pth --bits 8 6 4 3 2 --pgd-iter 20 \
    --out results/brl_e2_ce.json
```

### 4.4 E3 几何机制 + E3b 因果干预（表 tab:e3 / tab:e3i）
```bash
# 几何量 (ID/curv) 随位宽
python brl_e3.py --ckpt ckpt/brl_rn18_at.pth --bits 32 8 6 4 3 2 --feat-layer layer4 \
    --out results/brl_e3_at.json
# 因果干预：注入匹配量化误差的噪声
python brl_e3_interv.py --ckpt ckpt/brl_rn18_at.pth --bits 4 3 --feat-layer layer4 \
    --out results/brl_e3_interv_at.json
```

### 4.5 E4 Robust-QAT 三范式（表 tab:e4）
```bash
python brl_e4.py --ckpt ckpt/brl_rn18_at.pth --bits 4 3 2 --at-iters 8 \
    --out results/brl_e4.json
```

### 4.6 E4b 位宽专用 Robust-QAT / 二值基线（表 tab:qat）
```bash
# INT2 专用 QAT
python brl_binary_qat.py --mode qat --bits 2 --out ckpt/brl_rn18_qat_2.pth
# INT1 专用 QAT
python brl_binary_qat.py --mode qat --bits 1 --out ckpt/brl_rn18_qat_1.pth
# 扫描评估
brl_scan.py --ckpt ckpt/brl_rn18_qat_2.pth --model resnet18 --dataset cifar10 \
    --bits 32 8 4 3 2 1 --out results/qat_2_pgd20.json
brl_scan.py --ckpt ckpt/brl_rn18_qat_1.pth --model resnet18 --dataset cifar10 \
    --bits 32 8 4 3 2 1 --out results/qat_1_pgd20.json
```

### 4.7 E5 混合精度（表 tab:e5）
```bash
python brl_e5.py --ckpt ckpt/brl_rn18_at.pth --pgd-iter 20 --out results/brl_e5.json
```

### 4.8 E6 能量 Pareto frontier（表 tab:e6）
```bash
python brl_e6.py --ckpt ckpt/brl_rn18_at.pth --out results/brl_e6.json
```

### 4.9 E7 泛化验证
#### E7a 跨架构 — MobileNetV2（表 tab:e7a）
```bash
python brl_train.py --mode at --model mobilenetv2 --dataset cifar10 --epochs 80 \
    --out ckpt/brl_mv2_at.pth
python brl_scan.py --ckpt ckpt/brl_mv2_at.pth --model mobilenetv2 --dataset cifar10 \
    --bits 32 16 8 6 4 3 2 1 --out results/mv2_at_pgd20.json
```

#### E7b 跨数据集 — CIFAR-100（表 tab:e7b）
```bash
python brl_train.py --mode at --model resnet18 --dataset cifar100 --epochs 80 \
    --out ckpt/brl_rn18_c100_at.pth
python brl_scan.py --ckpt ckpt/brl_rn18_c100_at.pth --model resnet18 --dataset cifar100 \
    --bits 32 16 8 6 4 3 2 1 --out results/c100_at_pgd20.json
```

#### E7c 跨架构 — DeiT-Tiny（attention，表 tab:e7c，PGD-10）
```bash
python brl_train_vit.py --mode at --model deit_tiny --epochs 80 --lr 0.05 \
    --out ckpt/brl_deit_at.pth
python brl_scan_vit.py --ckpt ckpt/brl_deit_at.pth --model deit_tiny --dataset cifar10 \
    --bits 32 16 8 6 4 3 2 1 --out results/deit_at_pgd20.json
```
> E7c 使用 PGD-10（非 PGD-20），因 position-encoding 敏感性使 PGD-20 在该分辨率代价过高；
> 论文只比较相对景观，不跨攻击强度直接比对绝对值。

### 4.10 故障注入（表 tab:bfa）
```bash
python brl_bfa.py --ckpt ckpt/brl_rn18_at.pth --out results/bfa_at.json
python brl_bfa.py --ckpt ckpt/brl_rn18_ce.pth --out results/bfa_ce.json
```

### 4.11 ε-扫描（表 tab:eps，验证理论模型预测）
```bash
for eps in 4/255 8/255 16/255; do
  python brl_scan.py --ckpt ckpt/brl_rn18_at.pth --model resnet18 --dataset cifar10 \
      --bits 8 6 5 4 3 --pgd-iter 20 --eps $eps --out results/eps_${eps//\//_}.json
done
```

---

## 5. 论文表格 ↔ 命令 映射速查

| 论文表 | 实验 | 主要命令 |
|--------|------|---------|
| `tab:e1` | 主景观（4-seed） | `brl_train`×4 + `brl_scan` + mean±std |
| `tab:aa` | AutoAttack | `brl_autoattack_eval --limit 10000` |
| `tab:e2w/e2r` | 权重/激活机制、舍入 | `brl_e2` |
| `tab:e3` / `tab:e3i` | 几何 ID/curv、因果干预 | `brl_e3` / `brl_e3_interv` |
| `tab:e4` | Robust-QAT 三范式 | `brl_e4` |
| `tab:qat` | 位宽专用 QAT | `brl_binary_qat` + `brl_scan` |
| `tab:e5` | 混合精度敏感性 | `brl_e5` |
| `tab:e6` | 能量 Pareto | `brl_e6` |
| `tab:e7a` | MV2 跨架构 | `brl_train/scan --model mobilenetv2` |
| `tab:e7b` | CIFAR-100 跨数据集 | `brl_train/scan --dataset cifar100` |
| `tab:e7c` | DeiT-Tiny attention | `brl_train_vit/scan_vit --model deit_tiny` |
| `tab:bfa` | bit-flip 故障 | `brl_bfa` |
| `tab:eps` | ε-扫描 | `brl_scan --eps {4,8,16}/255` |
| `fig:landscape/geometry/pareto` | 论文三图 | `make_figs.py` |

---

## 6. 结果输出与验证

每个脚本在 `results/` 下生成 `.json`。论文所有表格数值均直接来自这些 JSON，未做人工修饰——
可复现性即"重跑命令 → 得到与论文一致的 JSON → 填入表格"。

若干反直觉结论已明确诚实标注（非实验误差）：
- *孤立激活量化的"高鲁棒"是 PGD 梯度失效伪影*，非真实鲁棒提升（E2）；
- *INT2 崩塌为真实现象*，已在多攻击预算（PGD-8/20、AutoAttack、L2）下交叉验证。

---

## 7. 复现注意事项

- **计算量**：完整复现全部实验（含 4-seed 训练 + 全部位宽谱 + QAT + 泛化）约需
  **数天 GPU 时间**（RTX 3090）。如仅验证结论，可先跑冒烟 + E1 单 seed 景观 + E2 归因。
- **随机性**：训练用 `--seed` 固定；评估（扫描/归因）默认 seed=0，保证可重复。
- **DeiT-Tiny**：`timm.create_model('deit_tiny_patch16_224', img_size=32, patch_size=4)`，
  需联网下载预训练配置（`pretrained=False`，仅结构）。
- **lada 几何管线**：`lada/manifold_analysis.py` 随包提供；脚本运行时自动优先使用同目录的 `lada/`。

---

## 8. License & Citation

代码仅供学术复现。引用请见论文参考文献；如有疑问，联系通讯作者
（投稿匿名版不含作者信息，正式版见单位页脚）。

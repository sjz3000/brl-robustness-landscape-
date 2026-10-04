"""
方向A · 二值/超低位宽鲁棒基线 (brl_binary_qat.py)
====================================================
目的: 检验"针对低位宽专门训练的鲁棒方法(QAT, 二值化/Binary Robust QAT)"能否
恢复 E1 中 direct PTQ 在 INT2/INT1 处的崩塌, 为 RA-BNN 式二值鲁棒方法提供诚实基线对照。

方法:
  - 训练期间对每个 Conv2d/Linear 权重做 fake-quant(STE, 见 brl_quant.fake_quant_ste),
    使模型"量化为 bits 位"仍保持有效梯度 -> 这是二值化/超低位宽 QAT 的标准做法。
  - 训练目标沿用模式 at(PGD-AT) 或 ce。默认 at, 因为 RA-BNN 类方法都是鲁棒训练。
  - 训练出的 QAT 模型部署在 bits 位: 评估时再 PTQ 该权重到 {bits, bits-1, 4} 全谱, 看低 bit 鲁棒。

对比基线: E1 tab:e1 中 INT2/INT1 直接 PTQ(无专门训练) = 崩塌(~6-10%)。
本实验 QAT-bits 应显著高于该崩塌水平, 才是"专门方法可恢复低位宽鲁棒"的证据。

用法:
    # 2-bit QAT 鲁棒训练 (AT, PGD-20)
    python brl_binary_qat.py --mode at --bits 2 --epochs 80 --out ckpt/brl_rn18_qat2.pth --log brl_qat2.log
    # 1-bit QAT 鲁棒训练
    python brl_binary_qat.py --mode at --bits 1 --epochs 80 --out ckpt/brl_rn18_qat1.pth --log brl_qat1.log
依赖: torch, torchvision, brl_quant, brl_train(get_loaders/build_model/set_seed/pgd_attack/evaluate)
"""
from __future__ import annotations
import argparse, time, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision
from brl_quant import fake_quant_ste, quantize_tensor
from brl_train import set_seed, get_loaders, build_model, _norm_pixel_bounds, pgd_attack, evaluate


class QuantConv2d(nn.Conv2d):
    """STE quantized conv (weight only)."""
    def __init__(self, c: nn.Conv2d, bits: int):
        super().__init__(c.in_channels, c.out_channels, c.kernel_size, c.stride,
                         c.padding, c.dilation, c.groups, c.bias is not None)
        self.bits = bits
        self.weight.data = c.weight.data.clone()
        if c.bias is not None:
            self.bias.data = c.bias.data.clone()
    def forward(self, x):
        wq = fake_quant_ste(self.weight, self.bits, scheme="sym", per_channel=True, ch_dim=0)
        return F.conv2d(x, wq, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QuantLinear(nn.Linear):
    def __init__(self, l: nn.Linear, bits: int):
        super().__init__(l.in_features, l.out_features, l.bias is not None)
        self.bits = bits
        self.weight.data = l.weight.data.clone()
        if l.bias is not None:
            self.bias.data = l.bias.data.clone()
    def forward(self, x):
        wq = fake_quant_ste(self.weight, self.bits, scheme="sym", per_channel=True, ch_dim=0)
        return F.linear(x, wq, self.bias)


def _replace(model: nn.Module, name: str, q: nn.Module):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    # key 可能是 str(普通子模块)或 int(Sequential 内元素)
    key = parts[-1]
    parent._modules[key if isinstance(key, int) else key] = q


def wrap_qat(model: nn.Module, bits: int) -> nn.Module:
    """把 model 中 Conv2d/Linear 原位替换为 STE 量化版本(保留 BN/ReLU/pool/bias)。"""
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.Conv2d):
            q = QuantConv2d(m, bits)
            _replace(model, name, q)
        elif isinstance(m, nn.Linear):
            q = QuantLinear(m, bits)
            _replace(model, name, q)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ce","at"], default="at")
    ap.add_argument("--bits", type=int, default=2, help="QAT deployment bitwidth (2 or 1)")
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--eps", type=float, default=8/255)
    ap.add_argument("--at-iters", type=int, default=20)
    ap.add_argument("--at-step-frac", type=float, default=2.0/8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ckpt/brl_rn18_qat.pth")
    ap.add_argument("--log", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true", help="1 epoch quick test")
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    logf = open(args.log, "w") if args.log else None
    def log(*a):
        m = " ".join(str(x) for x in a); print(m, flush=True)
        if logf: logf.write(m+"\n"); logf.flush()

    num_classes = 100 if args.dataset == "cifar100" else 10
    train_loader, test_loader = get_loaders(args.data_root, args.batch_size, dataset=args.dataset)
    base = build_model(args.model, num_classes).to(device)
    model = wrap_qat(base, args.bits).to(device)

    opt = optim_ = None
    import torch.optim as optim
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.wd, nesterov=True)
    def lr_at(ep):
        return 0.5*(1+np.cos(np.pi*ep/args.epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    criterion = nn.CrossEntropyLoss()

    epochs = 1 if args.smoke else args.epochs
    log(f"[brl_qat mode={args.mode} bits={args.bits} model={args.model}] device={device} epochs={epochs} seed={args.seed}")
    best_rob = best_clean = -1.0
    t0 = time.time()
    for ep in range(1, epochs+1):
        model.train(); running = n_b = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            if args.mode == "at":
                model.eval()
                with torch.enable_grad():
                    adv = pgd_attack(model, images, labels, eps=args.eps,
                                     step=args.eps*args.at_step_frac, iters=args.at_iters)
                model.train()
                loss = criterion(model(adv), labels)
            else:
                loss = criterion(model(images), labels)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item(); n_b += 1
        sched.step()
        if ep % 5 == 0 or ep == epochs:
            clean, robust = evaluate(model, test_loader, device, eps=args.eps,
                                     pgd_iters=args.at_iters, eval_adv=(args.mode=="at"))
            tag = ""
            if args.mode=="at" and robust > best_rob:
                best_rob = robust; torch.save(model.state_dict(), args.out); tag = "  <saved best>"
            elif args.mode=="ce" and clean > best_clean:
                best_clean = clean; torch.save(model.state_dict(), args.out); tag = "  <saved best>"
            log(f"ep={ep:>3}/{epochs} loss={running/n_b:.4f} clean={clean:.2f}% robust={robust:.2f}% "
                f"elapsed={(time.time()-t0)/60:.1f}min{tag}")
    torch.save(model.state_dict(), args.out)
    log(f"DONE. bits={args.bits} best_clean={best_clean:.2f}% best_robust={best_rob:.2f}% final={args.out}")
    if logf: logf.close()

if __name__ == "__main__":
    main()

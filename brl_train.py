"""
方向A · E1 骨干训练 (brl_train.py)
==================================
在边缘视觉骨干(这里 ResNet-18)上训练 CIFAR-10 基线模型, 供 BRLL(位宽-鲁棒景观)主扫描使用。

两种训练模式:
  --mode ce   : 标准交叉熵训练 (clean 高, 鲁棒弱)   -> 对应【部署最常见模型】
  --mode at   : PGD-AT 对抗训练 (鲁棒高, clean 略降)  -> 对应【安全敏感模型】

复用标准超参(固定 seed, 逐配置独立可复现)。

用法:
    # CE 骨干 (约 95% clean)
    python brl_train.py --mode ce  --epochs 200 --batch-size 128 --out ckpt/brl_rn18_ce.pth --log brl_train_ce.log
    # AT 骨干 (PGD-AT, eps=8/255, 10 iter; 约 83% clean / 45% robust)
    python brl_train.py --mode at  --epochs 80  --batch-size 128 --out ckpt/brl_rn18_at.pth --log brl_train_at.log

依赖: torch, torchvision。
"""
from __future__ import annotations
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_loaders(data_root: str, batch_size: int, num_workers: int = 4,
               dataset: str = "cifar10"):
    """支持 CIFAR-10 / CIFAR-100(相同归一化)。"""
    ds_cls = (torchvision.datasets.CIFAR100 if dataset == "cifar100"
              else torchvision.datasets.CIFAR10)
    tr = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    te = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_ds = ds_cls(root=data_root, train=True, download=True, transform=tr)
    test_ds = ds_cls(root=data_root, train=False, download=True, transform=te)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=512, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    """支持 E7 跨架构: ResNet-18 / MobileNetV2(深度可分离CNN)。"""
    if name == "resnet18":
        return torchvision.models.resnet18(num_classes=num_classes)
    if name == "mobilenetv2":
        return torchvision.models.mobilenet_v2(num_classes=num_classes)
    raise ValueError(f"unknown model {name}")


def build_resnet18(num_classes: int = 10) -> nn.Module:
    return torchvision.models.resnet18(num_classes=num_classes)


def _norm_pixel_bounds(device):
    """CIFAR-10 归一化后像素的有效范围 per-channel (C,1,1), 用于 PGD 投影到有效像素域。"""
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(3, 1, 1)
    return (-mean / std, (1 - mean) / std)


def pgd_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
               eps: float, step: float, iters: int):
    low, high = _norm_pixel_bounds(x.device)
    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(iters):
        out = model(x + delta)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()
        g = delta.grad.data
        delta.data = (delta.data + step * g.sign()).clamp(-eps, eps)
        delta.data = ((delta.data + x).clamp(low, high)) - x
        delta.grad.zero_()
    return x + delta


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, eps=8 / 255, pgd_iters=10,
             eval_adv: bool = True):
    model.eval()
    correct_c = correct_r = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        correct_c += (out.argmax(1) == labels).sum().item()
        if eval_adv:
            with torch.enable_grad():
                adv = pgd_attack(model, images, labels, eps=eps,
                                 step=eps * (2 / 8), iters=pgd_iters)
            out_adv = model(adv)
            correct_r += (out_adv.argmax(1) == labels).sum().item()
        total += images.size(0)
    clean = 100.0 * correct_c / total
    robust = 100.0 * correct_r / total if eval_adv else float("nan")
    model.train()
    return clean, robust


def main():
    ap = argparse.ArgumentParser(description="BRL E1 骨干训练")
    ap.add_argument("--mode", choices=["ce", "at"], default="ce")
    ap.add_argument("--model", default="resnet18",
                    help="resnet18 | mobilenetv2 (E7 跨架构)")
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"],
                    help="cifar10 | cifar100 (E7b 跨数据集)")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--eps", type=float, default=8 / 255)
    ap.add_argument("--at-iters", type=int, default=20)
    ap.add_argument("--at-step-frac", type=float, default=2.0 / 8.0)
    ap.add_argument("--schedule", type=str, default="cosine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ckpt/brl_rn18.pth")
    ap.add_argument("--log", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    logf = open(args.log, "w") if args.log else None
    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        if logf:
            logf.write(msg + "\n")
            logf.flush()

    num_classes = 100 if args.dataset == "cifar100" else 10
    train_loader, test_loader = get_loaders(args.data_root, args.batch_size,
                                            dataset=args.dataset)
    model = build_model(args.model, num_classes).to(device)
    print(f"[train] model={args.model} dataset={args.dataset} num_classes={num_classes}", flush=True)
    opt = optim.SGD(model.parameters(), lr=args.lr,
                    momentum=args.momentum, weight_decay=args.wd, nesterov=True)

    def lr_at(epoch):
        if args.schedule == "cosine":
            return 0.5 * (1 + np.cos(np.pi * epoch / args.epochs))
        # step decay: 0.1 at 50%/75%
        if epoch < 0.5 * args.epochs:
            return 1.0
        if epoch < 0.75 * args.epochs:
            return 0.1
        return 0.01

    sched = optim.lr_scheduler.LambdaLR(opt, lr_at)
    criterion = nn.CrossEntropyLoss()

    log(f"[brl_train mode={args.mode} model={args.model}] device={device} "
        f"epochs={args.epochs} seed={args.seed} lr={args.lr} wd={args.wd}")
    best_rob = -1.0
    best_clean = -1.0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_b = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            if args.mode == "at":
                model.eval()
                with torch.enable_grad():
                    adv = pgd_attack(model, images, labels, eps=args.eps,
                                     step=args.eps * args.at_step_frac,
                                     iters=args.at_iters)
                model.train()
                loss = criterion(model(adv), labels)
            else:
                loss = criterion(model(images), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            n_b += 1
        sched.step()
        if ep % 5 == 0 or ep == args.epochs:
            clean, robust = evaluate(
                model, test_loader, device, eps=args.eps,
                pgd_iters=args.at_iters, eval_adv=(args.mode == "at"))
            tag = ""
            if args.mode == "at" and robust > best_rob:
                best_rob = robust
                torch.save(model.state_dict(), args.out)
                tag = "  <saved best-robust>"
            elif args.mode == "ce" and clean > best_clean:
                best_clean = clean
                torch.save(model.state_dict(), args.out)
                tag = "  <saved best-clean>"
            log(f"ep={ep:>3}/{args.epochs} loss={running / n_b:.4f} "
                f"clean={clean:.2f}% robust={robust:.2f}% "
                f"elapsed={(time.time() - t0) / 60:.1f}min{tag}")
    # final save
    torch.save(model.state_dict(), args.out)
    log(f"DONE. best_clean={best_clean:.2f}% best_robust={best_rob:.2f}% "
        f"final_ckpt={args.out}")
    if logf:
        logf.close()


if __name__ == "__main__":
    main()

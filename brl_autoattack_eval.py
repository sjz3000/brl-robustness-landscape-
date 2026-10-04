#!/usr/bin/env python3
"""BRL AutoAttack + L2 独立评估脚本 (P0优先级)
用法:
    python3 brl_autoattack_eval.py --ckpt ckpt/brl_rn18_at.pth --dataset cifar10 \
        --bits 32 4 3 2 --norm L2 --eps 0.5 --limit 1000 --out results/brl_at_l2.json
"""
from __future__ import annotations
import argparse, copy, json, os, sys, time
import numpy as np
import torch, torchvision, torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brl_quant import ptq_quantize_weights, set_fp16, BITWIDTHS


# ── dataset ──────────────────────────────────────────────────────────────────────
def load_dataset(name, data_root, limit=None):
    tr = T.Compose([T.ToTensor(),
                    T.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
    ds_cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
    ds = ds_cls(root=data_root, train=False, download=True, transform=tr)
    if limit:
        ds = torch.utils.data.Subset(ds, np.arange(min(limit, len(ds))))
    return ds


# ── model ───────────────────────────────────────────────────────────────────────
def build_model(name, num_classes):
    if name == "resnet18":
        return torchvision.models.resnet18(num_classes=num_classes)
    raise ValueError(f"unknown model {name}")


# ── main ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--bits", nargs="+", default=["32","4","3","2"])
    parser.add_argument("--norm", default="Linf", choices=["Linf","L2"])
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--out", default="results/brl_autoattack.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bs", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bits = [int(b) for b in args.bits]

    # load dataset
    ds = load_dataset(args.dataset, args.data_root, limit=args.limit)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=2)
    num_classes = 100 if args.dataset == "cifar100" else 10

    # build model & load checkpoint
    model = build_model(args.model, num_classes)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt, strict=False)
    model.to(device).eval()

    results = {}
    base_model = copy.deepcopy(model).to(device)

    from autoattack import AutoAttack

    for bw in bits:
        t0 = time.time()
        m = copy.deepcopy(base_model).to(device)
        # apply PTQ
        if bw != 32:
            ptq_quantize_weights(m, bw)
        if bw == 16:
            set_fp16(m)
        m.eval()

        # clean accuracy
        wdtype = next(m.parameters()).dtype
        correct_c = total = 0
        for images, labels in loader:
            images = images.to(device=device, dtype=wdtype)
            labels = labels.to(device)
            out = m(images)
            correct_c += (out.argmax(1) == labels).sum().item()
            total += images.size(0)
        clean = 100.0 * correct_c / total if total else 0.0

        # AutoAttack / L2 PGD
        if args.norm == "Linf":
            adversary = AutoAttack(m, norm="Linf", eps=args.eps, version="standard")
            x_all, y_all = [], []
            for images, labels in loader:
                images = images.to(device=device, dtype=wdtype)
                x_all.append(images.cpu()); y_all.append(labels.cpu())
            x_tensor = torch.cat(x_all)[:args.limit]
            y_tensor = torch.cat(y_all)[:args.limit]
            # run AutoAttack
            with torch.enable_grad():
                x_adv = adversary.run_standard_evaluation(x_tensor, y_tensor, bs=args.bs)
            x_adv = x_adv.to(device=device, dtype=wdtype)
            y_tensor = y_tensor.to(device)
            correct_r = 0
            for i in range(0, len(x_adv), args.bs):
                batch_x = x_adv[i:i+args.bs]
                batch_y = y_tensor[i:i+args.bs]
                out = m(batch_x)
                correct_r += (out.argmax(1) == batch_y).sum().item()
            robust = 100.0 * correct_r / len(y_tensor) if len(y_tensor) else 0.0
        else:  # L2
            adversary = AutoAttack(m, norm="L2", eps=args.eps, version="standard")
            x_all, y_all = [], []
            for images, labels in loader:
                images = images.to(device=device, dtype=wdtype)
                x_all.append(images.cpu()); y_all.append(labels.cpu())
            x_tensor = torch.cat(x_all)[:args.limit]
            y_tensor = torch.cat(y_all)[:args.limit]
            with torch.enable_grad():
                x_adv = adversary.run_standard_evaluation(x_tensor, y_tensor, bs=args.bs)
            x_adv = x_adv.to(device=device, dtype=wdtype)
            y_tensor = y_tensor.to(device)
            correct_r = 0
            for i in range(0, len(x_adv), args.bs):
                batch_x = x_adv[i:i+args.bs]
                batch_y = y_tensor[i:i+args.bs]
                out = m(batch_x)
                correct_r += (out.argmax(1) == batch_y).sum().item()
            robust = 100.0 * correct_r / len(y_tensor) if len(y_tensor) else 0.0

        dt = time.time() - t0
        results[str(bw)] = {"clean": round(clean, 2), "robust": round(robust, 2), "norm": args.norm, "eps": args.eps, "time_s": round(dt, 1)}
        print(f"[bw={bw}] clean={clean:.2f}% robust={robust:.2f}% {args.norm}(eps={args.eps}) [{dt:.0f}s]")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"✅ saved → {args.out}")


if __name__ == "__main__":
    main()

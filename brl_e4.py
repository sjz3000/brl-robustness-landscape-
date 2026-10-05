"""
E4 Robust-QAT three-paradigm comparison (brl_e4.py)
==============================================
Compares three edge-friendly robust re-training (Robust-QAT) paradigms at low bitwidths (INT4/INT3/INT2),
testing how to preserve/improve adversarial robustness at low bitwidths (RQ3).

Three paradigms:
  F1  AT -> PTQ            : direct PTQ of a full-precision AT backbone (no retraining). Baseline, similar to E1.
  F2  PTQ -> retrain-AT    : initialize from a full-precision AT backbone, quantization-aware (STE w+a) + PGD-AT fine-tune 30 ep.
  F3  Joint QAT + AT       : random init, quantization-aware adversarial training from scratch (~20 ep short comparison).

Weight QAT: inject STE fake-quant (per-channel) via a forward hook during training.
Activation QAT: replace ReLU to inject STE activation quantization.
Evaluation: restore forward, then evaluate clean + PGD-20 after in-place PTQ weight quantization.
Depends on: brl_quant.py, brl_train.py (same directory).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brl_train as BT
import brl_quant as BQ


def _pc_scale(w, bits):
    qmax = 2 ** (bits - 1) - 1
    return w.detach().abs().amax(dim=tuple(range(1, w.dim())), keepdim=True) / qmax


def _q_conv(mm, x):
    wq = BQ.fake_quant_ste(mm.weight, mm.bits, "sym", True)
    if isinstance(mm, nn.Conv2d):
        return F.conv2d(x, wq, mm.bias, mm.stride, mm.padding, mm.dilation, mm.groups)
    if isinstance(mm, nn.Conv1d):
        return F.conv1d(x, wq, mm.bias, mm.stride, mm.padding, mm.dilation, mm.groups)
    return F.linear(x, wq, mm.bias)


def inject_weight_qat(model, bits):
    """Replace conv/linear forward with an STE weight-quantized version. Returns a restore closure."""
    from types import MethodType
    saved = {}
    for m in model.modules():
        if isinstance(m, BQ.MODULE_TYPES):
            saved[m] = m.forward
            def _mk(mm, of):
                def _wq_fwd(self, x):
                    return _q_conv(mm, x)
                return MethodType(_wq_fwd, mm)
            m.forward = _mk(m, m.forward)
            m.bits = bits
    def restore():
        for m, of in saved.items():
            m.forward = of
            if hasattr(m, "bits"):
                del m.bits
    return restore


def inject_act_qat(model, bits):
    """Replace ReLU with an STE activation-quantized version. Returns a restore."""
    from types import MethodType
    saved = {}
    for m in model.modules():
        if isinstance(m, (nn.ReLU,)):
            saved[m] = m.forward
            def _mk(mm, of, b):
                def _a_fwd(self, x):
                    out = of(x)
                    scale = out.detach().abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / (2**(b-1)-1)
                    q = torch.round(out / scale) * scale
                    return out + (q - out).detach()  # STE
                return MethodType(_a_fwd, mm)
            m.forward = _mk(m, m.forward, bits)
    def restore():
        for m, of in saved.items():
            m.forward = of
    return restore


def eval_quantized(model, device, test_loader, args, bits):
    """In-place weight-quantized clean+PGD evaluation (after restoring all hooks). Returns (clean, robust)."""
    model.eval()
    backup = BQ.make_weight_backup(model)
    BQ.ptq_quantize_weights(model, bits, scheme="sym", per_channel=True)
    correct_c = correct_r = total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            correct_c += (model(images).argmax(1) == labels).sum().item()
        with torch.enable_grad():
            adv = BT.pgd_attack(model, images, labels, eps=args.eps,
                                step=args.eps * 2 / 8, iters=args.at_iters)
        with torch.no_grad():
            correct_r += (model(adv).argmax(1) == labels).sum().item()
        total += images.size(0)
    BQ.restore_weights(model, backup)
    return 100.0 * correct_c / total, 100.0 * correct_r / total


def run_qat_from(model, device, train_loader, args, epochs, bits, lr, seed, tag):
    """Run quantization-aware adversarial training starting from model. Returns (final_model, restore_w, restore_a).
    restore_w/a are the hooks injected during training and must be restored before evaluation."""
    restore_w = inject_weight_qat(model, bits)
    restore_a = inject_act_qat(model, bits)
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n_ep = 1 if args.dry else epochs
    t0 = time.time()
    for ep in range(n_ep):
        run = nb = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            adv = BT.pgd_attack(model, images, labels, eps=args.eps,
                                step=args.eps * 2 / 8, iters=args.at_iters)
            opt.zero_grad()
            loss = F.cross_entropy(model(adv), labels)
            loss.backward()
            opt.step()
            run += loss.item(); nb += 1
        sched.step()
        if ep % 5 == 0 or ep == n_ep - 1:
            print(f"  {tag} bits={bits} ep={ep+1}/{n_ep} loss={run/max(nb,1):.4f} "
                  f"({(time.time()-t0)/60:.1f}min)", flush=True)
    return model, restore_w, restore_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/brl_rn18_at.pth")
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 3, 2])
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--epochs-f2", type=int, default=30)
    ap.add_argument("--epochs-f3", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr-f2", type=float, default=0.01)
    ap.add_argument("--lr-f3", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=8 / 255)
    ap.add_argument("--at-iters", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="brl_e4.json")
    ap.add_argument("--log", default=None)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    BT.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logf = open(args.log, "w") if args.log else None
    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        if logf:
            logf.write(msg + "\n"); logf.flush()

    train_loader, test_loader = BT.get_loaders(args.data_root, args.batch_size)
    results = {"meta": {"bits": args.bits, "ckpt": args.ckpt,
                        "epochs_f2": args.epochs_f2, "epochs_f3": args.epochs_f3,
                        "at_iters": args.at_iters, "seed": args.seed},
               "F1_at_ptq": {}, "F2_retrain_at": {}, "F3_joint": {}}

    # ---------- F1: AT -> PTQ ----------
    log("=== F1: AT->PTQ (direct quantization of a full-precision AT backbone, baseline) ===")
    base = BT.build_resnet18(10).to(device)
    base.load_state_dict(torch.load(args.ckpt, map_location=device))
    for b in args.bits:
        c, r = eval_quantized(base, device, test_loader, args, b)
        log(f"  F1 bits={b}: clean={c:.2f}% robust={r:.2f}%")
        results["F1_at_ptq"][str(b)] = {"clean": c, "robust": r}
    del base; torch.cuda.empty_cache()

    # ---------- F2: PTQ -> retrain AT (QAT, initialized from the AT backbone) ----------
    log("=== F2: QAT+retrain-AT (init from AT backbone, STE quantization + PGD-AT fine-tune) ===")
    for b in args.bits:
        m = BT.build_resnet18(10).to(device)
        m.load_state_dict(torch.load(args.ckpt, map_location=device))
        m, rw, ra = run_qat_from(m, device, train_loader, args, args.epochs_f2, b,
                                 args.lr_f2, args.seed, "F2")
        rw(); ra(); torch.cuda.empty_cache()  # restore hooks
        c, r = eval_quantized(m, device, test_loader, args, b)
        log(f"  F2 bits={b}: clean={c:.2f}% robust={r:.2f}%")
        results["F2_retrain_at"][str(b)] = {"clean": c, "robust": r}
        torch.save(m.state_dict(), f"ckpt/brl_e4_F2_b{b}.pth")
        del m; torch.cuda.empty_cache()

    # ---------- F3: Joint QAT + AT (from scratch) ----------
    log("=== F3: Joint QAT+AT (random init, quantization-aware adversarial training from scratch) ===")
    for b in args.bits:
        m = BT.build_resnet18(10).to(device)
        m, rw, ra = run_qat_from(m, device, train_loader, args, args.epochs_f3, b,
                                 args.lr_f3, args.seed, "F3")
        rw(); ra(); torch.cuda.empty_cache()
        c, r = eval_quantized(m, device, test_loader, args, b)
        log(f"  F3 bits={b}: clean={c:.2f}% robust={r:.2f}%")
        results["F3_joint"][str(b)] = {"clean": c, "robust": r}
        torch.save(m.state_dict(), f"ckpt/brl_e4_F3_b{b}.pth")
        del m; torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Saved -> {args.out}")
    if logf:
        logf.close()


if __name__ == "__main__":
    main()

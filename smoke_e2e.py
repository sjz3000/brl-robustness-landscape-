"""
P0 端到端自检: 用合成数据验证 BRL 冒烟管线逻辑(不需 CIFAR 下载)。
验证点: build_model / ptq_quantize_weights / pgd_attack / evaluate 全链路。
在GPU 上跑真数据时只需重新调用 brl_smoke.py。
"""
import os, sys, torch, numpy as np, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brl_quant import ptq_quantize_weights, restore_weights
import brl_smoke as B
from brl_quant import count_params as cnt_params


@torch.no_grad()
def main():
    torch.manual_seed(0); np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} (GPU for FP16; CPU skips bits=16)")
    model = B.build_model("resnet18", 10, pretrained=False).to(device).eval()
    print(f"model={model.__class__.__name__} params={cnt_params(model)/1e6:.1f}M")

    # 合成数据: 40 张随机 32x32 图像 + 伪标签
    x = torch.randn(40, 3, 32, 32).to(device)
    y = torch.randint(0, 10, (40,)).to(device)
    print(f"synthetic input: {x.shape} labels={y.shape}")

    for bits in [32, 16, 8, 4, 2]:
        if bits == 16 and device == "cpu":
            print("  bits= 16  skipped (CPU lacks fp16 conv)")
            continue
        if bits == 16:
            m = B.copy_deep(model).to(torch.float16)
            xx = x.half()
        else:
            m = B.copy_deep(model)
            m, _ = ptq_quantize_weights(m, bits, "sym", True)
            xx = x
        # clean 前向
        with torch.no_grad():
            out = m(xx)
            acc = (out.argmax(1).float() == y.float()).float().mean().item()
        # PGD 攻击(随机初始化模型下仅验证链路通, 数值无意义)
        with torch.enable_grad():
            adv = B.pgd_attack(m, xx, y, eps=8/255, step=2/255, iters=3)
        with torch.no_grad():
            out_a = m(adv)
            acc_a = (out_a.argmax(1).float() == y.float()).float().mean().item()
        print(f"  bits={bits:>3}  fwd_ok acc={acc:.2f}  pgd_ok robust={acc_a:.2f}  adv_shape={tuple(adv.shape)}")

    print("\nP0 END-TO-END SMOKE OK (synthetic)")

if __name__ == "__main__":
    main()

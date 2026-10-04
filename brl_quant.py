"""
方向A · P0 统一量化包装器 (brl_quant.py)
==========================================
面向"位宽-鲁棒景观(BRL)"实验的统一量化工具库。

设计目标
--------
1. 支持任意位宽谱 {FP32, FP16, INT8, INT6, INT4, INT3, INT2, INT1}(自定义 fake-quant,
   不受 torch.ao.quantization 仅 INT8 的限制)。
2. 支持 PTQ(训练后量化,权重就地替换,冒烟/景观扫描用) 与 QAT(STE 可导, 后续 E4 用)。
3. per-channel / per-tensor, 对称/非对称, 权重/激活可独立控制(支撑 E2 归因)。
4. 与数据/攻击解耦, 可被 brl_smoke.py 及各后续实验导入。

仅依赖 PyTorch。默认自动选设备(cuda/cpu)。
"""
from __future__ import annotations
import copy
import types
import torch
import torch.nn as nn

__all__ = [
    "BITWIDTHS", "quantize_tensor", "fake_quant_ste", "ptq_quantize_weights",
    "quantize_activation_inplace", "restore_weights", "set_fp16",
    "make_weight_backup", "MODULE_TYPES",
]

#: 一区连续位宽谱(含超低位宽)
BITWIDTHS = [32, 16, 8, 6, 4, 3, 2]

#: 需要处理权重的模块类型(卷积/线性)
MODULE_TYPES = (nn.Conv2d, nn.Conv1d, nn.Linear)


def _device_for(t: torch.Tensor) -> torch.device:
    return t.device


def quantize_tensor(
    x: torch.Tensor,
    bits: int,
    scheme: str = "sym",
    per_channel: bool = True,
    ch_dim: int = 0,
    clip: float = None,
) -> torch.Tensor:
    """均匀量化-反量化(确定性, 用于 PTQ)。

    返回与 x 同形状的"量化后重建张量", 数值为浮点但仅含 2^bits 个离散取值。
    - scheme="sym":  对称 [-qmax, qmax], zp=0, scale=max(|x|)/qmax
    - scheme="asym": 非对称 [min,max], 有 zero-point
    - per_channel=True: 沿 ch_dim 逐通道 scale/zero-point(ch_dim=0 适配权重)
    - clip: 可选的截断百分位(0~1), 抑制离群点(如 clip=0.99)
    """
    if bits >= 32:
        return x
    if bits == 16:
        return x.half().float()
    if bits == 1 and scheme == "sym":
        # INT1 对称二值化: 每通道幅度 amp = max|w|(沿除去 ch_dim 外的维度), 权重 -> sign(w)*amp
        max_dims = tuple(d for d in range(x.dim()) if d != ch_dim)
        if per_channel:
            amp = x.abs().amax(dim=max_dims, keepdim=True).clamp_min(1e-8)
        else:
            amp = x.abs().amax().clamp_min(1e-8)
        return (torch.where(x >= 0, 1.0, -1.0) * amp).type_as(x)
    qmax = float(2 ** (bits - 1) - 1) if scheme == "sym" else float(2 ** bits - 1)

    orig_shape = x.shape
    if per_channel:
        xr = x.reshape(x.shape[0], -1)  # (C, rest)
    else:
        xr = x.reshape(1, -1)

    if clip is not None:
        with torch.no_grad():
            hi = torch.quantile(xr.abs().reshape(xr.shape[0], -1), clip, dim=1)
            xr = torch.clamp(xr, -hi.unsqueeze(1), hi.unsqueeze(1))

    if scheme == "sym":
        amax = xr.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        scale = amax / qmax
        q = (xr / scale).round().clamp(-qmax, qmax)
        dq = q * scale
    else:  # asym
        lo = xr.amin(dim=1, keepdim=True)
        hi = xr.amax(dim=1, keepdim=True)
        scale = ((hi - lo) / qmax).clamp_min(1e-8)
        zp = (lo / scale).round()
        q = (xr / scale).round() - zp
        q = q.clamp(0, qmax)
        dq = (q + zp) * scale

    dq = dq.reshape(orig_shape)
    return dq


def fake_quant_ste(
    x: torch.Tensor,
    bits: int,
    scheme: str = "sym",
    per_channel: bool = True,
    ch_dim: int = 0,
) -> torch.Tensor:
    """straight-through estimator 的可导 fake-quant (QAT 用)。

    forward 返回量化重建值; backward 梯度近似直通(把量化视为恒等)。
    """
    if bits >= 32:
        return x
    xq = quantize_tensor(x, bits, scheme=scheme, per_channel=per_channel, ch_dim=ch_dim)
    return x + (xq - x).detach()


def set_fp16(model: nn.Module):
    """FP16 推理(不改变参数, 仅 half)。"""
    model.to(torch.float16)
    model.eval()
    return model


def make_weight_backup(model: nn.Module) -> dict:
    """备份所有 MODULE_TYPES 的权重, 返回 {module: original_weight_tensor}。"""
    backup = {}
    for m in model.modules():
        if isinstance(m, MODULE_TYPES) and hasattr(m, "weight"):
            backup[m] = m.weight.data.detach().clone()
    return backup


def ptq_quantize_weights(
    model: nn.Module,
    bits: int,
    scheme: str = "sym",
    per_channel: bool = True,
    inplace: bool = True,
):
    """PTQ: 就地(或返回副本)把 conv/linear 权重替换为量化重建值。

    返回 (model, backup)。注意: 就地修改会覆盖原有权重; 调用方需用 restore_weights 恢复。
    """
    if not inplace:
        model = copy.deepcopy(model)
    backup = make_weight_backup(model)
    for m, w in backup.items():
        with torch.no_grad():
            m.weight.data = quantize_tensor(
                w, bits, scheme=scheme, per_channel=per_channel, ch_dim=0
            )
    return model, backup


def restore_weights(model: nn.Module, backup: dict):
    """恢复 ptq_quantize_weights 备份的原始权重。"""
    for m, w in backup.items():
        with torch.no_grad():
            m.weight.data = w


def quantize_activation_inplace(
    model: nn.Module,
    bits: int,
    scheme: str = "sym",
    per_channel: bool = False,
    modules: tuple = (nn.ReLU,),
):
    """就地替换 ReLU(等激活) 的 forward, 在输出上做确定性激活量化(评估用, 不做 STE)。

    用于 E2 归因: 权重不变, 仅看激活量化的影响。通过注册临时 module-level hook 实现。
    返回可调用的 restore 函数。
    """
    handles = []
    for m in model.modules():
        if isinstance(m, modules):
            orig_fwd = m.forward

            def _new_fwd(_self, inp, _bits=bits, _scheme=scheme, _pc=per_channel,
                         _of=orig_fwd):
                out = _of(inp)
                return quantize_tensor(out, _bits, scheme=_scheme, per_channel=_pc, ch_dim=1)

            m.forward = types.MethodType(_new_fwd, m)
            handles.append((m, orig_fwd))
    def _restore():
        for m, of in handles:
            m.forward = of
    return _restore


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # 单元自检: 确认包装器基本功能正确
    torch.manual_seed(0)
    x = torch.randn(64, 3, 32, 32)
    conv = nn.Conv2d(3, 16, 3, padding=1)
    print("orig mean abs:", conv.weight.abs().mean().item())
    for b in [32, 16, 8, 4, 2]:
        qb = quantize_tensor(conv.weight.detach(), b, "sym", True)
        diff = (qb - conv.weight.detach()).abs().mean().item()
        n_uniq = qb.unique().numel()
        print(f"bits={b:>3}  reconstruction MAE={diff:.6f}  unique vals={n_uniq}")
    # QAT 梯度可导检查
    xg = torch.randn(64, 3, 32, 32, requires_grad=True)
    y = fake_quant_ste(xg, 4, "sym", False)
    loss = y.sum()
    loss.backward()
    assert xg.grad is not None and xg.grad.abs().sum().item() > 0
    print("QAT grad ok, x.grad norm:", xg.grad.norm().item())
    print("P0 quant wrapper self-test OK")

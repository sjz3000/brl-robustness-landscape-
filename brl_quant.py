"""
Unified quantization wrapper for the BRL (bitwidth-robustness landscape) experiments.
=========================================================================================
A unified quantization utility library for the bitwidth-robustness landscape (BRL) study.

Design goals
------------
1. Support an arbitrary bitwidth spectrum {FP32, FP16, INT8, INT6, INT4, INT3, INT2, INT1}
   with custom fake quantization, not limited to INT8 as in torch.ao.quantization.
2. Support PTQ (post-training quantization, in-place weight replacement, for smoke tests and
   landscape scans) and QAT (STE-differentiable, used by E4).
3. Independent per-channel / per-tensor, symmetric / asymmetric, and weight / activation
   control (enabling E2 attribution).
4. Decoupled from data and attacks; importable by brl_smoke.py and all downstream experiments.

Depends only on PyTorch. The device (cuda/cpu) is chosen automatically by default.
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

#: Continuous bitwidth spectrum including ultra-low bitwidths
BITWIDTHS = [32, 16, 8, 6, 4, 3, 2]

#: Module types whose weights are quantized (conv / linear)
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
    """Deterministic uniform quantize-dequantize (for PTQ).

    Returns a reconstructed tensor with the same shape as x; values are float but take only
    2^bits discrete levels.
    - scheme="sym":  symmetric [-qmax, qmax], zp=0, scale=max(|x|)/qmax
    - scheme="asym": asymmetric [min,max], with zero-point
    - per_channel=True: per-channel scale/zero-point along ch_dim (ch_dim=0 fits weights)
    - clip: optional truncation percentile (0~1) to suppress outliers (e.g. clip=0.99)
    """
    if bits >= 32:
        return x
    if bits == 16:
        return x.half().float()
    if bits == 1 and scheme == "sym":
        # INT1 symmetric binarization: per-channel magnitude amp = max|w| (over dims except ch_dim), then sign(w)*amp
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
    """Differentiable fake-quantization via the straight-through estimator (for QAT).

    forward returns the reconstructed quantized value; backward approximates the gradient
    as if quantization were the identity.
    """
    if bits >= 32:
        return x
    xq = quantize_tensor(x, bits, scheme=scheme, per_channel=per_channel, ch_dim=ch_dim)
    return x + (xq - x).detach()


def set_fp16(model: nn.Module):
    """FP16 inference (parameters unchanged, only half)."""
    model.to(torch.float16)
    model.eval()
    return model


def make_weight_backup(model: nn.Module) -> dict:
    """Back up MODULE_TYPES weights; returns {module: original_weight_tensor}."""
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
    """PTQ: replace conv/linear weights with reconstructed quantized values in place (or on a copy).

    Returns (model, backup). Note: in-place modification overwrites the original weights;
    the caller must use restore_weights to recover them.
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
    """Restore the original weights backed up by ptq_quantize_weights."""
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
    """Replace the forward of ReLU (etc.) in place, applying deterministic activation
    quantization to the output (for evaluation only, no STE).

    Used by E2 attribution: weights unchanged, only activation quantization is evaluated.
    Implemented via temporary module-level monkey-patching. Returns a callable restore.
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
    # Unit self-test: verify the wrapper's basic functionality
    torch.manual_seed(0)
    x = torch.randn(64, 3, 32, 32)
    conv = nn.Conv2d(3, 16, 3, padding=1)
    print("orig mean abs:", conv.weight.abs().mean().item())
    for b in [32, 16, 8, 4, 2]:
        qb = quantize_tensor(conv.weight.detach(), b, "sym", True)
        diff = (qb - conv.weight.detach()).abs().mean().item()
        n_uniq = qb.unique().numel()
        print(f"bits={b:>3}  reconstruction MAE={diff:.6f}  unique vals={n_uniq}")
    # QAT gradient differentiability check
    xg = torch.randn(64, 3, 32, 32, requires_grad=True)
    y = fake_quant_ste(xg, 4, "sym", False)
    loss = y.sum()
    loss.backward()
    assert xg.grad is not None and xg.grad.abs().sum().item() > 0
    print("QAT grad ok, x.grad norm:", xg.grad.norm().item())
    print("P0 quant wrapper self-test OK")

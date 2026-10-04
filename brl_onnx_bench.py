#!/usr/bin/env python3
"""BRL ONNX 真实推理延迟基准 (P0-3, 边缘部署硬件验证的前置演示)
对比 FP32 vs dynamic-INT8 在 onnxruntime CPU 上的单样本推理延迟。
诚实报告 ResNet-18 (卷积为主) 在通用 CPU 动态量化下的加速情况。
"""
import time, numpy as np, torch, torchvision
import onnxruntime as ort
import onnxruntime.quantization as quant

ckpt_path = "ckpt/brl_rn18_at.pth"
m = torchvision.models.resnet18(num_classes=10)
ckpt = torch.load(ckpt_path, map_location="cpu")
if "model_state_dict" in ckpt: ckpt = ckpt["model_state_dict"]
m.load_state_dict(ckpt, strict=False)
m.eval()

x = torch.randn(1, 3, 32, 32)
torch.onnx.export(m, x, "/tmp/rn18_fp32.onnx",
                  input_names=["input"], output_names=["logits"], opset_version=12)

# dynamic INT8 (QUInt8) - 量化 Linear/MatMul/Gemm 权重
quant.quantize_dynamic("/tmp/rn18_fp32.onnx", "/tmp/rn18_int8_qu8.onnx",
                       weight_type=quant.QuantType.QUInt8)

def bench(path, n=300, label=""):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    xnp = np.random.rand(1, 3, 32, 32).astype(np.float32)
    for _ in range(20):
        sess.run(None, {"input": xnp})
    t0 = time.time()
    for _ in range(n):
        sess.run(None, {"input": xnp})
    ms = (time.time() - t0) / n * 1000
    return ms

fp32 = bench("/tmp/rn18_fp32.onnx", label="fp32")
int8 = bench("/tmp/rn18_int8_qu8.onnx", label="int8")
print(f"FP32: {fp32:.4f} ms/iter")
print(f"INT8(dynamic,QU8): {int8:.4f} ms/iter")
print(f"speedup: {fp32/int8:.2f}x  (conv-heavy ResNet-18: dynamic-INT8 主要加速 FC/MatMul，Conv 未加速)")

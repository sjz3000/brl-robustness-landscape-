#!/usr/bin/env python3
"""BRL ONNX real-inference latency benchmark
Compares single-sample inference latency of FP32 vs dynamic-INT8 on onnxruntime CPU.
Honestly reports the speedup of ResNet-18 (conv-dominated) under dynamic quantization on a general CPU.
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

# dynamic INT8 (QUInt8) - quantize Linear/MatMul/Gemm weights
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
print(f"speedup: {fp32/int8:.2f}x  (conv-heavy ResNet-18: dynamic-INT8 mainly accelerates FC/MatMul, Conv is not accelerated)")

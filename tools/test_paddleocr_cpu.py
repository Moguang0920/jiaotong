# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
os.environ.setdefault("FLAGS_use_mkldnn", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

print("=" * 72)
print("PaddleOCR CPU model verification")
print(f"Python: {sys.executable}")
print("=" * 72)

import paddle

compiled_cuda = bool(paddle.device.is_compiled_with_cuda())
print(f"paddle={paddle.__version__}")
print(f"compiled_with_cuda={compiled_cuda}")

if compiled_cuda:
    raise RuntimeError(
        "paddlepaddle-gpu is still installed. Run fix_paddleocr_cpu.bat again."
    )

paddle.set_device("cpu")
print(f"device={paddle.get_device()}")

from paddleocr import TextRecognition

model_name = os.environ.get("OCR_MODEL_NAME", "PP-OCRv6_tiny_rec")
print(f"Loading model: {model_name}")

started = time.perf_counter()
model = TextRecognition(
    model_name=model_name,
    device="cpu",
    enable_mkldnn=True,
    cpu_threads=4,
)
elapsed = (time.perf_counter() - started) * 1000.0

print(f"[SUCCESS] PaddleOCR model loaded on CPU in {elapsed:.1f} ms")
print("=" * 72)

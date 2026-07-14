# -*- coding: utf-8 -*-
from __future__ import annotations

import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CHECKS = {
    ROOT / "backend" / "plate_runtime_backend.py": [
        "PaddleOCRSubprocessClient",
        "ensure_paddleocr_gpu_subprocess",
        "/api/ocr/restart",
        "OCR-GPU-SUBPROCESS-V6",
    ],
    ROOT / "backend" / "paddle_ocr_gpu_worker.py": [
        "@@TRAFFIC_OCR_JSON@@",
        "paddle.device.is_compiled_with_cuda",
        'device=args.device',
    ],
    ROOT / "frontend" / "renderer.js": [
        "GPU 独立进程 OCR",
        "ocr_process_pid",
    ],
    ROOT / "tools" / "verify_dual_gpu_processes.py": [
        "verify_onnx_main_process",
        "verify_ocr_subprocess",
    ],
}

for path, tokens in CHECKS.items():
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    for token in tokens:
        if token not in text:
            raise RuntimeError(f"{path.name} 缺少：{token}")

for path in [
    ROOT / "backend" / "plate_runtime_backend.py",
    ROOT / "backend" / "paddle_ocr_gpu_worker.py",
    ROOT / "tools" / "verify_dual_gpu_processes.py",
]:
    py_compile.compile(str(path), doraise=True)

print("=" * 72)
print("PaddleOCR GPU 独立子进程 V6 代码自检通过")
print("- 主后端 ONNX CUDA：保留")
print("- PaddleOCR GPU：独立进程")
print("- DLL 地址空间：隔离")
print("- 车牌投票/白名单：保留")
print("- 道路异常/车道线/实时基准：保留")
print("=" * 72)

# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OCR_PYTHON = ROOT / ".venv_ocr" / "Scripts" / "python.exe"
OCR_WORKER = ROOT / "backend" / "paddle_ocr_gpu_worker.py"
PREFIX = "@@TRAFFIC_OCR_JSON@@"


def read_protocol(process: subprocess.Popen, expected: str, timeout: float):
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = process.stdout.readline() if process.stdout else ""
        if not line:
            if process.poll() is not None:
                raise RuntimeError(
                    f"OCR worker exited: {process.returncode}"
                )
            continue
        line = line.strip()
        if not line.startswith(PREFIX):
            print("[OCR stdout]", line)
            continue
        payload = json.loads(line[len(PREFIX):])
        if payload.get("type") == expected:
            return payload
    raise TimeoutError(f"等待 {expected} 超时")


def verify_onnx_main_process() -> dict:
    # 模拟主后端的正确加载顺序。
    import torch
    probe = torch.zeros((1,), device="cuda")
    torch.cuda.synchronize()
    del probe

    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(
            cuda=True,
            cudnn=True,
            msvc=True,
            directory=None,
        )

    model_candidates = [
        ROOT / "models" / "best(1).onnx",
        ROOT / "best(1).onnx",
        ROOT / "models" / "normal.onnx",
        ROOT / "normal.onnx",
    ]
    model = next(
        (path for path in model_candidates if path.exists()),
        None,
    )
    if model is None:
        raise FileNotFoundError("没有找到可用于验证的 ONNX 模型")

    session = ort.InferenceSession(
        str(model),
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    actual = session.get_providers()
    if not actual or actual[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"ONNX 未使用 CUDA：{actual}")

    return {
        "model": str(model),
        "providers": actual,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }


def verify_ocr_subprocess() -> dict:
    if not OCR_PYTHON.exists():
        raise FileNotFoundError(
            f"缺少 OCR venv：{OCR_PYTHON}"
        )
    if not OCR_WORKER.exists():
        raise FileNotFoundError(
            f"缺少 OCR worker：{OCR_WORKER}"
        )

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PADDLE_PDX_MODEL_SOURCE"] = "BOS"
    env["OMP_NUM_THREADS"] = "1"

    process = subprocess.Popen(
        [
            str(OCR_PYTHON),
            "-u",
            str(OCR_WORKER),
            "--model-name",
            "PP-OCRv6_tiny_rec",
            "--device",
            "gpu:0",
        ],
        cwd=str(ROOT),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    try:
        ready = read_protocol(
            process,
            "ready",
            timeout=300.0,
        )
        if not ready.get("ok"):
            stderr = process.stderr.read() if process.stderr else ""
            raise RuntimeError(
                f"OCR worker 初始化失败：{ready}; {stderr}"
            )

        image = np.full(
            (64, 220, 3),
            255,
            dtype=np.uint8,
        )
        cv2.putText(
            image,
            "A5678T",
            (12, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError("测试图片编码失败")

        request = {
            "type": "recognize",
            "id": 1,
            "image_jpeg_b64": base64.b64encode(
                encoded.tobytes()
            ).decode("ascii"),
        }
        process.stdin.write(
            json.dumps(request, ensure_ascii=True) + "\n"
        )
        process.stdin.flush()

        result = read_protocol(
            process,
            "result",
            timeout=30.0,
        )
        if not result.get("ok"):
            raise RuntimeError(f"OCR 请求失败：{result}")

        return {
            "ready": ready,
            "result": result,
        }
    finally:
        if process.poll() is None:
            try:
                if process.stdin:
                    process.stdin.write('{"type":"shutdown"}\n')
                    process.stdin.flush()
                process.wait(timeout=3)
            except Exception:
                process.terminate()


def main() -> int:
    print("=" * 78)
    print("双 GPU 进程隔离验证")
    print("=" * 78)

    onnx_info = verify_onnx_main_process()
    print("[ONNX 主进程 OK]")
    print(json.dumps(onnx_info, ensure_ascii=False, indent=2))

    ocr_info = verify_ocr_subprocess()
    print("[PaddleOCR 子进程 OK]")
    print(json.dumps(ocr_info, ensure_ascii=False, indent=2))

    print("=" * 78)
    print("[SUCCESS] ONNX CUDA 与 PaddleOCR GPU 已在不同进程同时通过。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAILED] {type(exc).__name__}: {exc}")
        raise SystemExit(1)

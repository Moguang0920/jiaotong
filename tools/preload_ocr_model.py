# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OCR_PYTHON = ROOT / ".venv_ocr" / "Scripts" / "python.exe"
WORKER = ROOT / "backend" / "paddle_ocr_gpu_worker.py"
PREFIX = "@@TRAFFIC_OCR_JSON@@"


def main() -> int:
    if not OCR_PYTHON.exists():
        print(f"缺少 OCR 独立环境：{OCR_PYTHON}")
        print("请运行 setup_paddleocr_gpu_subprocess.bat")
        return 1
    if not WORKER.exists():
        print(f"缺少 OCR worker：{WORKER}")
        return 1

    model_name = os.environ.get(
        "TRAFFIC_OCR_MODEL",
        "PP-OCRv6_tiny_rec",
    )
    process = subprocess.Popen(
        [
            str(OCR_PYTHON),
            "-u",
            str(WORKER),
            "--model-name",
            model_name,
            "--device",
            "gpu:0",
        ],
        cwd=str(ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    try:
        deadline = time.time() + 300.0
        while time.time() < deadline:
            line = process.stdout.readline() if process.stdout else ""
            if not line:
                if process.poll() is not None:
                    print(f"OCR worker 已退出：{process.returncode}")
                    return 1
                continue
            line = line.strip()
            if not line.startswith(PREFIX):
                print(line)
                continue
            payload = json.loads(line[len(PREFIX):])
            if payload.get("type") == "ready":
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return 0 if payload.get("ok") else 1
        print("等待 OCR GPU 模型加载超时。")
        return 1
    finally:
        if process.poll() is None:
            try:
                if process.stdin:
                    process.stdin.write('{"type":"shutdown"}\n')
                    process.stdin.flush()
                process.wait(timeout=2)
            except Exception:
                process.terminate()


if __name__ == "__main__":
    raise SystemExit(main())

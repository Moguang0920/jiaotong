# -*- coding: utf-8 -*-
"""PaddleOCR GPU 独立子进程。

此文件只能由 plate_runtime_backend.py 或验证脚本启动。
进程内不导入 torch / onnxruntime，确保 PaddlePaddle GPU 使用自己的
CUDA/cuDNN DLL，不与主后端进程发生 WinError 127 冲突。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, Tuple

PROTOCOL_PREFIX = "@@TRAFFIC_OCR_JSON@@"


def configure_stdio() -> None:
    try:
        sys.stdout.reconfigure(
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
    except Exception:
        pass


def emit(payload: Dict[str, Any]) -> None:
    print(
        PROTOCOL_PREFIX
        + json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def log(message: str) -> None:
    print(f"[PaddleOCRWorker] {message}", file=sys.stderr, flush=True)


def extract_result(output: Any) -> Tuple[str, float]:
    if output is None:
        return "", 0.0

    if not isinstance(output, (list, tuple)):
        try:
            output = list(output)
        except Exception:
            output = [output]

    if not output:
        return "", 0.0

    obj = output[0]
    data = getattr(obj, "json", None)
    if callable(data):
        data = data()

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            data = {}

    if isinstance(data, dict):
        inner = data.get("res", data)
        if isinstance(inner, dict):
            if "rec_text" in inner:
                return (
                    str(inner.get("rec_text") or ""),
                    float(inner.get("rec_score") or 0.0),
                )
            if "rec_texts" in inner:
                texts = inner.get("rec_texts") or []
                scores = inner.get("rec_scores") or []
                text = "".join(str(item) for item in texts if item)
                score = float(max(scores)) if scores else 0.0
                return text, score

    if isinstance(obj, dict):
        inner = obj.get("res", obj)
        return (
            str(inner.get("rec_text") or ""),
            float(inner.get("rec_score") or 0.0),
        )

    return "", 0.0


def main() -> int:
    configure_stdio()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        default="PP-OCRv6_tiny_rec",
    )
    parser.add_argument(
        "--device",
        default="gpu:0",
    )
    args = parser.parse_args()

    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")

    try:
        import cv2
        import numpy as np
        import paddle

        if not bool(paddle.device.is_compiled_with_cuda()):
            raise RuntimeError(
                "OCR 专用环境安装的不是 paddlepaddle-gpu。"
            )

        device_count = int(paddle.device.cuda.device_count())
        if device_count < 1:
            raise RuntimeError(
                "PaddlePaddle GPU 已安装，但没有检测到可用 NVIDIA GPU。"
            )

        paddle.set_device(args.device)

        gpu_name = "NVIDIA GPU"
        try:
            gpu_name = str(
                paddle.device.cuda.get_device_name(0)
            )
        except Exception:
            pass

        from paddleocr import TextRecognition

        log(
            f"Loading {args.model_name} on {args.device}; "
            f"paddle={paddle.__version__}; gpu={gpu_name}"
        )

        started = time.perf_counter()
        model = TextRecognition(
            model_name=args.model_name,
            device=args.device,
        )
        load_ms = (
            time.perf_counter() - started
        ) * 1000.0

        emit({
            "type": "ready",
            "ok": True,
            "model_name": args.model_name,
            "device": args.device,
            "paddle_version": str(paddle.__version__),
            "gpu_name": gpu_name,
            "gpu_count": device_count,
            "model_load_ms": round(load_ms, 2),
            "python_executable": sys.executable,
            "pid": os.getpid(),
        })

        for raw_line in sys.stdin:
            line = str(raw_line or "").strip()
            if not line:
                continue

            request: Dict[str, Any] = {}
            try:
                request = json.loads(line)
                request_type = str(request.get("type") or "")

                if request_type == "shutdown":
                    emit({
                        "type": "shutdown_ack",
                        "ok": True,
                    })
                    return 0

                if request_type == "ping":
                    emit({
                        "type": "pong",
                        "ok": True,
                        "id": request.get("id"),
                    })
                    continue

                if request_type != "recognize":
                    emit({
                        "type": "error",
                        "ok": False,
                        "id": request.get("id"),
                        "error": f"未知请求类型：{request_type}",
                    })
                    continue

                request_id = int(request.get("id") or 0)
                encoded_text = str(
                    request.get("image_jpeg_b64") or ""
                )
                if not encoded_text:
                    raise ValueError("缺少 image_jpeg_b64")

                image_bytes = base64.b64decode(
                    encoded_text.encode("ascii"),
                    validate=True,
                )
                array = np.frombuffer(
                    image_bytes,
                    dtype=np.uint8,
                )
                image = cv2.imdecode(
                    array,
                    cv2.IMREAD_COLOR,
                )
                if image is None or image.size == 0:
                    raise ValueError("无法解码 OCR 图片")

                t0 = time.perf_counter()
                output = model.predict(
                    input=image,
                    batch_size=1,
                )
                ocr_ms = (
                    time.perf_counter() - t0
                ) * 1000.0
                text, confidence = extract_result(output)

                emit({
                    "type": "result",
                    "ok": True,
                    "id": request_id,
                    "text": text,
                    "confidence": float(confidence),
                    "ocr_ms": round(ocr_ms, 3),
                })

            except Exception as exc:
                emit({
                    "type": "error",
                    "ok": False,
                    "id": request.get("id"),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                log(traceback.format_exc())

        return 0

    except Exception as exc:
        emit({
            "type": "ready",
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "python_executable": sys.executable,
            "pid": os.getpid(),
        })
        log(traceback.format_exc())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

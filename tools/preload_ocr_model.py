# -*- coding: utf-8 -*-
from __future__ import annotations

import os


def main() -> int:
    try:
        from paddleocr import TextRecognition
    except Exception as exc:
        print(f"PaddleOCR 导入失败：{exc}")
        return 1

    model_name = os.environ.get("TRAFFIC_OCR_MODEL", "PP-OCRv6_tiny_rec")
    print(f"正在初始化并下载 OCR 模型：{model_name}")
    print("首次下载需要联网，下载完成后会保存在本机缓存中。")
    try:
        model = TextRecognition(model_name=model_name)
        print(f"OCR 模型初始化成功：{type(model).__name__}")
        return 0
    except Exception as exc:
        print(f"OCR 模型初始化失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "runtime_data" / "environment_report.json"


def command_version(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        return (result.stdout or result.stderr or "").strip()
    except Exception as exc:
        return f"不可用: {exc}"


def main() -> int:
    print("=" * 72)
    print("智慧交通视觉感知系统：环境检查")
    print("=" * 72)

    report: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "architecture": platform.architecture()[0],
        "platform": platform.platform(),
        "modules": {},
        "models": {},
        "node": {},
        "gpu": {},
    }
    errors: list[str] = []
    warnings: list[str] = []

    version_pair = sys.version_info[:2]
    if version_pair not in {(3, 12), (3, 13)}:
        errors.append(f"当前 Python 为 {sys.version.split()[0]}，应使用 3.12 或 3.13。")
    if sys.maxsize <= 2**32:
        errors.append("当前是 32 位 Python，必须使用 64 位 Python。")

    module_tests = {
        "numpy": "numpy",
        "opencv": "cv2",
        "fastapi": "fastapi",
        "pydantic": "pydantic",
        "Pillow": "PIL",
        "onnxruntime": "onnxruntime",
        "paddle": "paddle",
        "paddleocr": "paddleocr",
        "uvicorn": "uvicorn",
    }

    loaded = {}
    for label, module_name in module_tests.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            report["modules"][label] = {"ok": True, "version": str(version)}
            loaded[module_name] = module
            print(f"[OK] {label}: {version}")
        except Exception as exc:
            report["modules"][label] = {"ok": False, "error": repr(exc)}
            errors.append(f"{label} 导入失败：{exc}")
            print(f"[FAIL] {label}: {exc}")

    ort = loaded.get("onnxruntime")
    if ort is not None:
        try:
            providers = list(ort.get_available_providers())
            report["gpu"]["onnxruntime_providers"] = providers
            print(f"[ORT] Providers: {providers}")
            if "CUDAExecutionProvider" not in providers:
                warnings.append("ONNX Runtime 没有 CUDAExecutionProvider，将使用 CPU。")
        except Exception as exc:
            warnings.append(f"读取 ONNX Runtime Provider 失败：{exc}")

    paddle = loaded.get("paddle")
    if paddle is not None:
        try:
            compiled_cuda = bool(paddle.device.is_compiled_with_cuda())
            device_count = int(paddle.device.cuda.device_count()) if compiled_cuda else 0
            report["gpu"]["paddle_compiled_with_cuda"] = compiled_cuda
            report["gpu"]["paddle_cuda_device_count"] = device_count
            print(f"[Paddle] CUDA 编译: {compiled_cuda}, GPU 数量: {device_count}")
            if not compiled_cuda:
                warnings.append("PaddleOCR 使用 CPU；这不影响 ONNX YOLO 使用 GPU。")
        except Exception as exc:
            warnings.append(f"读取 Paddle GPU 状态失败：{exc}")

    try:
        from paddleocr import TextRecognition  # noqa: F401
        print("[OK] PaddleOCR TextRecognition API 可用")
    except Exception as exc:
        errors.append(f"PaddleOCR TextRecognition API 不可用：{exc}")

    model_groups = {
        "plate": ["best(1).onnx", "best.onnx"],
        "vehicle": ["hearmap.onnx", "heatmap.onnx"],
        "stop": ["stop.onnx"],
        "normal": ["normal.onnx"],
    }
    for model_key, names in model_groups.items():
        candidates = []
        for name in names:
            candidates.extend([ROOT / name, ROOT / "models" / name])
        existing = next((path for path in candidates if path.exists()), None)
        report["models"][model_key] = str(existing) if existing else None
        if existing:
            print(f"[MODEL OK] {model_key}: {existing}")
        else:
            warnings.append(
                f"缺少 {model_key} 模型，可选文件名：{', '.join(names)}；"
                f"请放到 {ROOT / 'models'}。"
            )

    node = shutil.which("node")
    npm = shutil.which("npm")
    if not node:
        common_node = Path(r"C:\Program Files\nodejs\node.exe")
        node = str(common_node) if common_node.exists() else None
    if not npm:
        common_npm = Path(r"C:\Program Files\nodejs\npm.cmd")
        npm = str(common_npm) if common_npm.exists() else None

    report["node"]["node"] = command_version([node, "--version"]) if node else "missing"
    report["node"]["npm"] = command_version([npm, "--version"]) if npm else "missing"
    print(f"[Node] {report['node']['node']}")
    print(f"[npm] {report['node']['npm']}")
    if not node or not npm:
        errors.append("Node.js/npm 不可用。")

    if not (ROOT / "node_modules" / "electron").exists():
        warnings.append("没有发现 node_modules/electron，请重新运行一键安装脚本。")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "-" * 72)
    if warnings:
        print("警告：")
        for item in warnings:
            print(" - " + item)
    if errors:
        print("错误：")
        for item in errors:
            print(" - " + item)

    print(f"\n检查报告：{REPORT_PATH}")
    if errors:
        print("环境检查未通过。")
        return 1

    print("环境检查通过。模型缺失警告不会阻止安装，但对应功能无法运行。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

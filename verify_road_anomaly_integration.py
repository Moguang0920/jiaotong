# -*- coding: utf-8 -*-
"""道路异常实时集成包自检。"""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    ROOT / "backend" / "plate_runtime_backend.py",
    ROOT / "backend" / "normal_lane_detector.py",
    ROOT / "backend" / "runtime_road_anomaly_detector.py",
    ROOT / "frontend" / "dashboard.html",
    ROOT / "frontend" / "renderer.js",
    ROOT / "frontend" / "styles.css",
]


def require_text(path: Path, token: str) -> None:
    text = path.read_text(encoding="utf-8")
    if token not in text:
        raise RuntimeError(f"{path.name} 缺少集成标记：{token}")


def main() -> int:
    for path in REQUIRED:
        if not path.exists():
            raise FileNotFoundError(f"缺少文件：{path}")

    for path in REQUIRED[:3]:
        py_compile.compile(str(path), doraise=True)

    require_text(REQUIRED[0], "RuntimeRoadAnomalyDetector")
    require_text(REQUIRED[0], "/api/normal/baseline/rebuild")
    require_text(REQUIRED[2], "live_baseline_multi_reference")
    require_text(REQUIRED[3], "rebuildNormalBaselineBtn")
    require_text(REQUIRED[4], "drawRoadAnomalyOverlay")
    require_text(REQUIRED[4], "points.length !== 4")

    sys.path.insert(0, str(ROOT))
    from backend.runtime_road_anomaly_detector import RuntimeRoadAnomalyDetector  # noqa: F401
    from backend.normal_lane_detector import NormalLaneDetector  # noqa: F401

    print("=" * 72)
    print("道路异常实时集成自检通过")
    print("- 四点 ROI：通过")
    print("- 原连续车道线模块：通过")
    print("- 当前视频源实时基准：通过")
    print("- 多基准道路异常模块：通过")
    print("- 前端异常框与重新采集按钮：通过")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[自检失败] {type(exc).__name__}: {exc}")
        raise SystemExit(1)

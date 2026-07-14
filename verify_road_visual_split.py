# -*- coding: utf-8 -*-
from pathlib import Path
import py_compile

ROOT = Path(__file__).resolve().parent
CHECKS = {
    ROOT / "frontend" / "dashboard.html": [
        'id="normalLaneCanvas"',
        'id="normalAnomalyMaskFeed"',
        '道路辅助视图',
    ],
    ROOT / "frontend" / "renderer.js": [
        'function drawNormalLaneAssist()',
        'function refreshNormalDebugPreview',
        'const boxes = isNormalDetectorSelected() ? []',
        '/api/normal/debug.png',
    ],
    ROOT / "frontend" / "styles.css": [
        '.normal-visual-diagnostics',
        '.normal-diagnostic-grid',
        '.normal-mask-stage',
    ],
    ROOT / "backend" / "runtime_road_anomaly_detector.py": [
        'def _update_debug_preview(',
        'def get_debug_preview_png(',
        'panel[voted_mask > 0] = (0, 0, 255)',
    ],
    ROOT / "backend" / "plate_runtime_backend.py": [
        '@app.get("/api/normal/debug.png")',
        'def api_normal_debug_png()',
    ],
}

errors = []
for path, markers in CHECKS.items():
    if not path.exists():
        errors.append(f"缺少文件：{path.relative_to(ROOT)}")
        continue
    text = path.read_text(encoding="utf-8")
    for marker in markers:
        if marker not in text:
            errors.append(f"{path.relative_to(ROOT)} 缺少标记：{marker}")

for rel in [
    Path("backend/runtime_road_anomaly_detector.py"),
    Path("backend/plate_runtime_backend.py"),
]:
    path = ROOT / rel
    if path.exists():
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            errors.append(f"Python 语法检查失败：{rel} -> {exc}")

if errors:
    print("验证失败：")
    for item in errors:
        print("-", item)
    raise SystemExit(1)

print("验证通过：主视频异常优先显示、下方车道线视图、蓝/红差分图均已接入。")

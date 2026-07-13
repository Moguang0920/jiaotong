# -*- coding: utf-8 -*-
"""
道路 ROI 手动标定工具

作用：
1. 默认打开：
   road_anomaly_data/camera_01/reference_bank/reference_01.png
2. 用鼠标沿道路“外边缘”依次点击，形成多边形。
3. 保存：
   road_anomaly_data/camera_01/road_roi/road_mask.png
   road_anomaly_data/camera_01/road_roi/road_roi_preview.png
   road_anomaly_data/camera_01/road_roi/road_roi.json

操作：
- 鼠标左键：添加顶点
- 鼠标右键：撤销一个顶点
- Backspace：撤销一个顶点
- R：清空重画
- Enter：确认并保存
- Esc：取消

注意：
- 框的是整块“可通行路面”，不是中间某一条车道线。
- 45° 透视视角下，远处应更窄、近处应更宽。
- 建议先避开斑马线、两侧人行道、建筑和树木。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


WINDOW_NAME = "Road ROI Calibration"
DEFAULT_IMAGE = (
    Path("road_anomaly_data")
    / "camera_01"
    / "reference_bank"
    / "reference_01.png"
)
OUTPUT_DIR = (
    Path("road_anomaly_data")
    / "camera_01"
    / "road_roi"
)

MAX_DISPLAY_WIDTH = 1500
MAX_DISPLAY_HEIGHT = 900


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"图片编码失败：{path}")
    encoded.tofile(str(path))


def resize_to_fit(
    image: np.ndarray,
    max_width: int,
    max_height: int,
) -> Tuple[np.ndarray, float]:
    height, width = image.shape[:2]

    scale = min(
        max_width / max(width, 1),
        max_height / max(height, 1),
        1.0,
    )

    if scale >= 1.0:
        return image.copy(), 1.0

    resized = cv2.resize(
        image,
        (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


class RoadROISelector:
    def __init__(self, image: np.ndarray):
        self.original = image
        self.preview, self.display_scale = resize_to_fit(
            image,
            MAX_DISPLAY_WIDTH,
            MAX_DISPLAY_HEIGHT,
        )
        self.points: List[Tuple[int, int]] = []
        self.confirmed = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((int(x), int(y)))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()

    def render(self) -> np.ndarray:
        canvas = self.preview.copy()

        # 轻微压暗，方便看清 ROI 高亮
        canvas = cv2.addWeighted(
            canvas,
            0.82,
            np.zeros_like(canvas),
            0.18,
            0,
        )

        if self.points:
            pts = np.array(
                self.points,
                dtype=np.int32,
            ).reshape((-1, 1, 2))

            if len(self.points) >= 3:
                layer = canvas.copy()

                cv2.fillPoly(
                    layer,
                    [pts],
                    (40, 210, 130),
                )

                canvas = cv2.addWeighted(
                    layer,
                    0.28,
                    canvas,
                    0.72,
                    0,
                )

                cv2.polylines(
                    canvas,
                    [pts],
                    True,
                    (0, 225, 255),
                    4,
                    cv2.LINE_AA,
                )

            elif len(self.points) >= 2:
                cv2.polylines(
                    canvas,
                    [pts],
                    False,
                    (0, 225, 255),
                    4,
                    cv2.LINE_AA,
                )

            for index, point in enumerate(self.points, start=1):
                cv2.circle(
                    canvas,
                    point,
                    8,
                    (0, 80, 255),
                    -1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    canvas,
                    str(index),
                    (point[0] + 10, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        # 顶部操作栏
        cv2.rectangle(
            canvas,
            (0, 0),
            (canvas.shape[1], 82),
            (10, 17, 27),
            -1,
        )

        cv2.putText(
            canvas,
            "Left click: add point | Right click / Backspace: undo | R: reset",
            (18, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"Enter: save | Esc: cancel | Points: {len(self.points)}",
            (18, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (80, 230, 255),
            2,
            cv2.LINE_AA,
        )

        return canvas

    def select(self) -> Optional[np.ndarray]:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

        while True:
            cv2.imshow(WINDOW_NAME, self.render())
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10):  # Enter
                if len(self.points) < 3:
                    print("至少需要 3 个顶点。")
                    continue

                self.confirmed = True
                break

            if key in (8, 127):  # Backspace
                if self.points:
                    self.points.pop()

            elif key in (ord("r"), ord("R")):
                self.points.clear()

            elif key == 27:  # Esc
                break

            try:
                if cv2.getWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_VISIBLE,
                ) < 1:
                    break
            except cv2.error:
                break

        cv2.destroyWindow(WINDOW_NAME)

        if not self.confirmed:
            return None

        height, width = self.original.shape[:2]
        original_points = []

        for x, y in self.points:
            original_x = int(round(x / self.display_scale))
            original_y = int(round(y / self.display_scale))

            original_x = max(0, min(width - 1, original_x))
            original_y = max(0, min(height - 1, original_y))

            original_points.append(
                [original_x, original_y]
            )

        return np.array(
            original_points,
            dtype=np.int32,
        )


def save_roi(
    image: np.ndarray,
    image_path: Path,
    roi_points: np.ndarray,
    output_dir: Path,
) -> None:
    height, width = image.shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    polygon = roi_points.reshape((-1, 1, 2))

    cv2.fillPoly(
        mask,
        [polygon],
        255,
    )

    preview = image.copy()
    layer = preview.copy()

    cv2.fillPoly(
        layer,
        [polygon],
        (40, 210, 130),
    )

    preview = cv2.addWeighted(
        layer,
        0.28,
        preview,
        0.72,
        0,
    )

    cv2.polylines(
        preview,
        [polygon],
        True,
        (0, 225, 255),
        4,
        cv2.LINE_AA,
    )

    for index, (x, y) in enumerate(
        roi_points.tolist(),
        start=1,
    ):
        cv2.circle(
            preview,
            (int(x), int(y)),
            8,
            (0, 80, 255),
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            preview,
            str(index),
            (int(x) + 10, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_path = output_dir / "road_mask.png"
    preview_path = output_dir / "road_roi_preview.png"
    json_path = output_dir / "road_roi.json"

    imwrite_unicode(mask_path, mask)
    imwrite_unicode(preview_path, preview)

    record = {
        "camera_id": "camera_01",
        "source_image": str(image_path).replace("\\", "/"),
        "image_width": width,
        "image_height": height,
        "point_count": int(len(roi_points)),
        "points": [
            {
                "x": int(x),
                "y": int(y),
            }
            for x, y in roi_points.tolist()
        ],
        "mask_file": "road_mask.png",
        "preview_file": "road_roi_preview.png",
        "meaning": (
            "白色区域为道路异常检测有效区域，"
            "黑色区域完全忽略。"
        ),
    }

    json_path.write_text(
        json.dumps(
            record,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("道路 ROI 保存完成")
    print(f"道路 Mask：{mask_path}")
    print(f"预览图片：{preview_path}")
    print(f"坐标配置：{json_path}")
    print("=" * 72)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    image_path = project_root / DEFAULT_IMAGE

    if not image_path.exists():
        raise FileNotFoundError(
            f"没有找到默认基准图：\n{image_path}\n"
            "请先运行多基准图片生成脚本，"
            "或检查 reference_01.png 是否存在。"
        )

    image = imread_unicode(image_path)

    if image is None:
        raise RuntimeError(
            f"无法读取图片：{image_path}"
        )

    print("=" * 72)
    print("道路 ROI 标定")
    print("框整块道路路面，不是框某一条车道线。")
    print("建议沿道路左右外边缘，按顺时针或逆时针点击。")
    print("=" * 72)

    roi_points = RoadROISelector(image).select()

    if roi_points is None:
        print("已取消，没有保存。")
        return

    save_roi(
        image=image,
        image_path=image_path,
        roi_points=roi_points,
        output_dir=project_root / OUTPUT_DIR,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        cv2.destroyAllWindows()
        print(
            f"[失败] {type(exc).__name__}: {exc}"
        )
        sys.exit(1)

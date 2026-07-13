# -*- coding: utf-8 -*-
"""
把已经标定好的道路 ROI 实时画到“正常道路.mp4”上。

用途：
1. 验证你画的道路区域是否准确。
2. 验证视频轻微晃动时，道路 ROI 能否跟着视频一起移动。
3. 这一步后续会直接复用到真正的道路障碍物检测链路里。

依赖前提：
你已经完成：
road_anomaly_data/camera_01/
├── reference_bank/reference_01.png
└── road_roi/road_roi.json

运行：
    python draw_road_roi_on_video.py

默认输入视频：
    正常道路.mp4

默认输出视频：
    road_anomaly_data/camera_01/road_roi_overlay.mp4

操作：
- Q / Esc：退出
- Space：暂停 / 继续
- S：单步前进一帧（暂停状态下）
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# 配置
# ============================================================

VIDEO_FILENAME = "正常道路.mp4"
CAMERA_ROOT = Path("road_anomaly_data") / "camera_01"
REFERENCE_IMAGE = CAMERA_ROOT / "reference_bank" / "reference_01.png"
ROAD_ROI_JSON = CAMERA_ROOT / "road_roi" / "road_roi.json"
OUTPUT_VIDEO = CAMERA_ROOT / "road_roi_overlay.mp4"

WINDOW_NAME = "Road ROI Overlay on Video"

# 是否保存带框视频
SAVE_OUTPUT_VIDEO = True

# 显示尺寸上限
MAX_DISPLAY_WIDTH = 1600
MAX_DISPLAY_HEIGHT = 920

# 为提高速度，不一定每帧都重新估计变换。
# 例如设为2，表示隔一帧重新匹配一次，其余帧复用上次结果。
PROCESS_EVERY_N_FRAMES = 1

# ORB 匹配参数
ORB_FEATURES = 7000
LOWE_RATIO = 0.76
MIN_GOOD_MATCHES = 24
MIN_INLIERS = 16
MIN_INLIER_RATIO = 0.25

# 允许的轻微晃动范围保护
MAX_TRANSLATION_X_RATIO = 0.25
MAX_TRANSLATION_Y_RATIO = 0.25
MIN_SCALE = 0.80
MAX_SCALE = 1.20
MAX_ROTATION_DEGREES = 14.0


# ============================================================
# 基础工具
# ============================================================

def imread_unicode(path: Path, flags: int) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


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


def open_video_with_fallback(
    video_path: Path,
) -> Tuple[cv2.VideoCapture, Optional[Path]]:
    """
    优先直接打开中文路径视频；
    如果当前 OpenCV/FFmpeg 不支持中文路径，则复制为临时英文文件名。
    """
    capture = cv2.VideoCapture(str(video_path))

    if capture.isOpened():
        return capture, None

    capture.release()

    temp_path = video_path.with_name("_normal_road_temp.mp4")
    print(
        "[提示] 当前 OpenCV 无法直接打开中文视频路径，"
        f"临时复制为：{temp_path.name}"
    )

    shutil.copy2(video_path, temp_path)

    capture = cv2.VideoCapture(str(temp_path))

    if not capture.isOpened():
        capture.release()
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"无法打开视频：{video_path}")

    return capture, temp_path


def load_roi_points(roi_json_path: Path) -> np.ndarray:
    data = json.loads(
        roi_json_path.read_text(encoding="utf-8")
    )

    points = data.get("points", [])

    if len(points) < 3:
        raise RuntimeError(
            f"road_roi.json 中顶点数量不足：{len(points)}"
        )

    polygon = np.array(
        [[int(item["x"]), int(item["y"])] for item in points],
        dtype=np.float32,
    )

    return polygon


def create_orb():
    return cv2.ORB_create(
        nfeatures=ORB_FEATURES,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=21,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=12,
    )


def normalize_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(gray)


def estimate_anchor_to_frame(
    anchor_image: np.ndarray,
    current_frame: np.ndarray,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    计算从 anchor 图坐标 -> 当前帧坐标 的仿射矩阵。
    """
    if anchor_image.shape[:2] != current_frame.shape[:2]:
        return None, {
            "status": "failed",
            "reason": "image_size_mismatch",
        }

    anchor_gray = normalize_gray(anchor_image)
    frame_gray = normalize_gray(current_frame)

    orb = create_orb()

    keypoints_anchor, descriptors_anchor = orb.detectAndCompute(
        anchor_gray,
        None,
    )
    keypoints_frame, descriptors_frame = orb.detectAndCompute(
        frame_gray,
        None,
    )

    if (
        descriptors_anchor is None
        or descriptors_frame is None
        or len(keypoints_anchor) < 10
        or len(keypoints_frame) < 10
    ):
        return None, {
            "status": "failed",
            "reason": "not_enough_keypoints",
            "anchor_keypoints": len(keypoints_anchor or []),
            "frame_keypoints": len(keypoints_frame or []),
        }

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False,
    )

    knn_matches = matcher.knnMatch(
        descriptors_anchor,
        descriptors_frame,
        k=2,
    )

    good_matches = []

    for pair in knn_matches:
        if len(pair) != 2:
            continue

        first, second = pair

        if first.distance < LOWE_RATIO * second.distance:
            good_matches.append(first)

    if len(good_matches) < MIN_GOOD_MATCHES:
        return None, {
            "status": "failed",
            "reason": "not_enough_good_matches",
            "good_matches": len(good_matches),
            "anchor_keypoints": len(keypoints_anchor),
            "frame_keypoints": len(keypoints_frame),
        }

    anchor_points = np.float32([
        keypoints_anchor[match.queryIdx].pt
        for match in good_matches
    ]).reshape(-1, 1, 2)

    frame_points = np.float32([
        keypoints_frame[match.trainIdx].pt
        for match in good_matches
    ]).reshape(-1, 1, 2)

    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        anchor_points,
        frame_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.5,
        maxIters=4000,
        confidence=0.995,
        refineIters=30,
    )

    if matrix is None or inlier_mask is None:
        return None, {
            "status": "failed",
            "reason": "affine_estimation_failed",
            "good_matches": len(good_matches),
        }

    inliers = int(inlier_mask.reshape(-1).sum())
    inlier_ratio = inliers / max(len(good_matches), 1)

    if (
        inliers < MIN_INLIERS
        or inlier_ratio < MIN_INLIER_RATIO
    ):
        return None, {
            "status": "failed",
            "reason": "weak_geometry_consistency",
            "good_matches": len(good_matches),
            "inliers": inliers,
            "inlier_ratio": round(float(inlier_ratio), 4),
        }

    a, b, tx = matrix[0]
    c, d, ty = matrix[1]

    scale_x = math.sqrt(a * a + c * c)
    scale_y = math.sqrt(b * b + d * d)
    average_scale = (scale_x + scale_y) / 2.0
    rotation_degrees = math.degrees(math.atan2(c, a))

    height, width = anchor_image.shape[:2]

    transform_ok = (
        abs(tx) <= width * MAX_TRANSLATION_X_RATIO
        and abs(ty) <= height * MAX_TRANSLATION_Y_RATIO
        and MIN_SCALE <= average_scale <= MAX_SCALE
        and abs(rotation_degrees) <= MAX_ROTATION_DEGREES
    )

    diagnostics = {
        "status": "success" if transform_ok else "failed",
        "reason": None if transform_ok else "transform_out_of_range",
        "good_matches": len(good_matches),
        "inliers": inliers,
        "inlier_ratio": round(float(inlier_ratio), 4),
        "translation_x": round(float(tx), 4),
        "translation_y": round(float(ty), 4),
        "scale": round(float(average_scale), 6),
        "rotation_degrees": round(float(rotation_degrees), 5),
    }

    if not transform_ok:
        return None, diagnostics

    return matrix.astype(np.float32), diagnostics


def transform_polygon(
    polygon: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    transformed = cv2.transform(
        polygon.reshape(-1, 1, 2),
        matrix,
    )

    return transformed.reshape(-1, 2)


def clip_polygon(
    polygon: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    clipped = polygon.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
    return clipped


def draw_overlay(
    frame: np.ndarray,
    polygon: np.ndarray,
    diagnostics_text: str,
    frame_index: int,
    total_frames: int,
    paused: bool,
    using_fallback: bool,
) -> np.ndarray:
    output = frame.copy()

    polygon_int = polygon.astype(np.int32).reshape((-1, 1, 2))

    layer = output.copy()
    cv2.fillPoly(
        layer,
        [polygon_int],
        (40, 210, 130),
    )

    output = cv2.addWeighted(
        layer,
        0.22,
        output,
        0.78,
        0,
    )

    cv2.polylines(
        output,
        [polygon_int],
        True,
        (0, 225, 255),
        4,
        cv2.LINE_AA,
    )

    # 顶点序号
    for idx, (x, y) in enumerate(polygon.astype(int), start=1):
        cv2.circle(
            output,
            (int(x), int(y)),
            7,
            (0, 80, 255),
            -1,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            str(idx),
            (int(x) + 9, int(y) - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 92),
        (10, 17, 27),
        -1,
    )

    state_text = "PAUSED" if paused else "PLAYING"
    fallback_text = " | fallback" if using_fallback else ""

    top_line = (
        f"{state_text}{fallback_text} | "
        f"Frame: {frame_index}/{total_frames if total_frames > 0 else '?'}"
    )

    cv2.putText(
        output,
        top_line,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        "Road ROI overlay (anchor ROI mapped to current frame)",
        (16, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (90, 230, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        diagnostics_text[:95],
        (16, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (220, 240, 220),
        1,
        cv2.LINE_AA,
    )

    return output


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    project_root = Path(__file__).resolve().parent

    video_path = project_root / VIDEO_FILENAME
    anchor_image_path = project_root / REFERENCE_IMAGE
    roi_json_path = project_root / ROAD_ROI_JSON
    output_video_path = project_root / OUTPUT_VIDEO

    if not video_path.exists():
        raise FileNotFoundError(
            f"没有找到视频：{video_path}"
        )

    if not anchor_image_path.exists():
        raise FileNotFoundError(
            f"没有找到主基准图：{anchor_image_path}"
        )

    if not roi_json_path.exists():
        raise FileNotFoundError(
            f"没有找到道路 ROI 配置：{roi_json_path}"
        )

    anchor_image = imread_unicode(
        anchor_image_path,
        cv2.IMREAD_COLOR,
    )

    if anchor_image is None:
        raise RuntimeError(
            f"无法读取主基准图：{anchor_image_path}"
        )

    anchor_polygon = load_roi_points(roi_json_path)

    capture, temp_video_path = open_video_with_fallback(video_path)

    writer = None

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(source_fps) or source_fps <= 1.0:
            source_fps = 25.0

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        success, first_frame = capture.read()
        if not success or first_frame is None:
            raise RuntimeError("无法读取视频第一帧。")

        # 后续所有视频帧都缩放到 anchor 图尺寸进行匹配和绘制
        anchor_h, anchor_w = anchor_image.shape[:2]
        first_frame_resized = cv2.resize(
            first_frame,
            (anchor_w, anchor_h),
            interpolation=cv2.INTER_AREA,
        )

        if SAVE_OUTPUT_VIDEO:
            output_video_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(output_video_path),
                fourcc,
                source_fps,
                (anchor_w, anchor_h),
            )

        print("=" * 78)
        print("道路 ROI 视频叠加演示")
        print(f"输入视频：{video_path}")
        print(f"主基准图：{anchor_image_path}")
        print(f"道路 ROI：{roi_json_path}")
        if SAVE_OUTPUT_VIDEO:
            print(f"输出视频：{output_video_path}")
        print("=" * 78)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        paused = False
        single_step = False

        frame_index = 1
        processed_counter = 0

        current_frame_resized = first_frame_resized
        last_polygon = anchor_polygon.copy()
        last_diagnostics_text = "anchor init"
        last_good_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )

        while True:
            should_read = (
                frame_index == 1
                or (not paused)
                or single_step
            )

            if should_read and frame_index > 1:
                success, frame = capture.read()
                if not success or frame is None:
                    print("视频播放结束。")
                    break

                current_frame_resized = cv2.resize(
                    frame,
                    (anchor_w, anchor_h),
                    interpolation=cv2.INTER_AREA,
                )

                processed_counter += 1

                using_fallback = False

                if (
                    processed_counter % PROCESS_EVERY_N_FRAMES == 0
                    or frame_index == 2
                ):
                    matrix, diagnostics = estimate_anchor_to_frame(
                        anchor_image,
                        current_frame_resized,
                    )

                    if matrix is not None:
                        transformed = transform_polygon(
                            anchor_polygon,
                            matrix,
                        )
                        transformed = clip_polygon(
                            transformed,
                            anchor_w,
                            anchor_h,
                        )

                        last_polygon = transformed
                        last_good_matrix = matrix

                        last_diagnostics_text = (
                            f"matches={diagnostics['good_matches']} | "
                            f"inliers={diagnostics['inliers']} | "
                            f"scale={diagnostics['scale']} | "
                            f"rot={diagnostics['rotation_degrees']} deg"
                        )
                    else:
                        # 当前帧匹配失败，就回退到上一次成功变换
                        transformed = transform_polygon(
                            anchor_polygon,
                            last_good_matrix,
                        )
                        transformed = clip_polygon(
                            transformed,
                            anchor_w,
                            anchor_h,
                        )

                        last_polygon = transformed
                        using_fallback = True

                        reason = diagnostics.get("reason", "unknown")
                        matches = diagnostics.get("good_matches", 0)
                        inliers = diagnostics.get("inliers", 0)

                        last_diagnostics_text = (
                            f"fallback | reason={reason} | "
                            f"matches={matches} | inliers={inliers}"
                        )
                else:
                    using_fallback = True

                single_step = False
            else:
                using_fallback = False

            display_frame = draw_overlay(
                frame=current_frame_resized,
                polygon=last_polygon,
                diagnostics_text=last_diagnostics_text,
                frame_index=frame_index,
                total_frames=total_frames,
                paused=paused,
                using_fallback=using_fallback,
            )

            display, _ = resize_to_fit(
                display_frame,
                MAX_DISPLAY_WIDTH,
                MAX_DISPLAY_HEIGHT,
            )

            cv2.imshow(WINDOW_NAME, display)

            if writer is not None:
                writer.write(display_frame)

            delay = max(1, int(round(1000.0 / source_fps))) if not paused else 30
            key = cv2.waitKey(delay)

            key_code = -1 if key == -1 else (key & 0xFF)

            if key_code in (ord("q"), ord("Q"), 27):
                break

            if key_code == ord(" "):
                paused = not paused

            elif key_code in (ord("s"), ord("S")):
                if paused:
                    single_step = True

            try:
                if cv2.getWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_VISIBLE,
                ) < 1:
                    break
            except cv2.error:
                break

            if not paused or single_step:
                frame_index += 1

    finally:
        capture.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

        if temp_video_path is not None:
            temp_video_path.unlink(missing_ok=True)

    print("=" * 78)
    print("处理完成")
    if SAVE_OUTPUT_VIDEO:
        print(f"输出视频已保存：{output_video_path}")
    print("=" * 78)
    print(
        "如果道路框能稳定跟着视频画面移动，"
        "下一步就可以接入车辆 YOLO，开始做“道路区域 - 车辆区域”的有效比较。"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        cv2.destroyAllWindows()
        print(f"[失败] {type(exc).__name__}: {exc}")
        sys.exit(1)

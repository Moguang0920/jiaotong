# -*- coding: utf-8 -*-
"""
从项目根目录的“正常道路.mp4”建立“多基准图片库”。

与旧版不同：
1. 不再把多帧做平均值或中位数合成，因此不会产生叠影背景。
2. 从正常道路视频中抽取大量候选帧。
3. 自动过滤严重模糊、过暗、过亮的帧。
4. 根据画面位置、轻微晃动、光照和道路外观进行聚类。
5. 每一类保存一张清晰且有代表性的原始帧，形成多个校验基准。

输出：
road_anomaly_data/camera_01/
├── reference_bank/
│   ├── reference_01.png
│   ├── reference_02.png
│   └── ...
├── reference_contact_sheet.jpg
└── reference_bank.json

后续实时检测逻辑：
当前帧 -> 从 reference_bank 中寻找最接近的基准图
       -> 对当前帧和该基准图做小范围配准
       -> 屏蔽车辆区域
       -> 检测剩余道路区域差异

运行：
    python generate_multi_road_references.py
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# 配置
# ============================================================

VIDEO_FILENAME = "正常道路.mp4"
OUTPUT_ROOT = Path("road_anomaly_data") / "camera_01"
REFERENCE_DIR_NAME = "reference_bank"

# 最终保存多少张正常道路基准图。
# 对一分钟视频，12～20张通常够用。
REFERENCE_COUNT = 16

# 最多读取多少张候选帧。
# 240张相当于一分钟视频平均每0.25秒检查一次。
MAX_CANDIDATE_FRAMES = 240

# 跳过视频开头和结尾，避免连接、停止录制时的异常帧。
EDGE_SKIP_SECONDS = 1.0

# 后续实时检测也应该统一缩放到这个最大宽度。
# 原视频宽度不超过该值时不会放大。
PROCESS_MAX_WIDTH = 1280

# 质量过滤参数。
# 最终还会根据视频自身分布自动计算模糊阈值。
ABSOLUTE_MIN_SHARPNESS = 18.0
MIN_BRIGHTNESS = 15.0
MAX_BRIGHTNESS = 242.0
MIN_CONTRAST = 8.0

# 聚类随机种子，保证重复运行结果较稳定。
RANDOM_SEED = 20260713


@dataclass
class CandidateFrame:
    frame_index: int
    timestamp_seconds: float
    image: np.ndarray
    feature: np.ndarray
    sharpness: float
    brightness: float
    contrast: float
    quality_score: float
    cluster_id: int = -1


# ============================================================
# 文件与视频辅助
# ============================================================

def imwrite_unicode(
    path: Path,
    image: np.ndarray,
    params: Optional[List[int]] = None,
) -> None:
    """兼容 Windows 中文路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"

    ok, encoded = cv2.imencode(
        suffix,
        image,
        params or [],
    )

    if not ok:
        raise RuntimeError(f"图片编码失败：{path}")

    encoded.tofile(str(path))


def resize_for_processing(
    frame: np.ndarray,
    max_width: int,
) -> Tuple[np.ndarray, float]:
    height, width = frame.shape[:2]

    if width <= max_width:
        return frame.copy(), 1.0

    scale = max_width / float(width)
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        frame,
        (max_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


def open_video_with_fallback(
    video_path: Path,
) -> Tuple[cv2.VideoCapture, Optional[Path]]:
    """
    优先直接打开中文视频路径。
    如果当前 OpenCV 无法读取中文路径，则临时复制为英文文件名。
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


# ============================================================
# 画面质量与特征
# ============================================================

def calculate_quality(
    frame: np.ndarray,
) -> Tuple[float, float, float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    brightness = float(gray.mean())
    contrast = float(gray.std())

    # 曝光越接近正常范围，惩罚越小。
    exposure_penalty = 0.0

    if brightness < 45.0:
        exposure_penalty += (45.0 - brightness) / 12.0

    if brightness > 210.0:
        exposure_penalty += (brightness - 210.0) / 12.0

    # 对清晰度取对数，防止极少数高噪声帧占据过大优势。
    quality_score = (
        math.log1p(max(sharpness, 0.0))
        + contrast * 0.015
        - exposure_penalty
    )

    return (
        sharpness,
        brightness,
        contrast,
        quality_score,
    )


def build_visual_feature(
    frame: np.ndarray,
) -> np.ndarray:
    """
    构建用于区分摄像头轻微位移、晃动和光照变化的特征。

    不进行图像配准，因为本步骤就是要保留不同晃动位置作为不同基准。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 保留整体布局与摄像头位移信息。
    small_gray = cv2.resize(
        gray,
        (32, 18),
        interpolation=cv2.INTER_AREA,
    )

    # 均衡化后保留道路结构，减弱纯亮度变化的影响。
    equalized = cv2.equalizeHist(small_gray)

    edges = cv2.Canny(
        equalized,
        45,
        130,
    )

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [16],
        [0, 256],
    ).reshape(-1)

    histogram = histogram / max(
        float(histogram.sum()),
        1.0,
    )

    feature = np.concatenate([
        equalized.reshape(-1).astype(np.float32) / 255.0,
        edges.reshape(-1).astype(np.float32) / 255.0 * 0.85,
        histogram.astype(np.float32) * 5.0,
    ])

    return feature.astype(np.float32)


# ============================================================
# 候选帧读取与筛选
# ============================================================

def build_sample_indexes(
    total_frames: int,
    fps: float,
) -> np.ndarray:
    if total_frames <= 0:
        raise RuntimeError("无法取得视频总帧数。")

    skip_frames = int(
        round(max(fps, 1.0) * EDGE_SKIP_SECONDS)
    )

    start_index = min(
        skip_frames,
        max(total_frames - 1, 0),
    )

    end_index = max(
        start_index,
        total_frames - skip_frames - 1,
    )

    available_frames = end_index - start_index + 1

    sample_count = min(
        MAX_CANDIDATE_FRAMES,
        available_frames,
    )

    return np.linspace(
        start_index,
        end_index,
        num=max(sample_count, 1),
        dtype=np.int64,
    )


def read_candidates(
    capture: cv2.VideoCapture,
    indexes: np.ndarray,
    fps: float,
) -> Tuple[List[CandidateFrame], Tuple[int, int], float]:
    raw_candidates: List[CandidateFrame] = []
    source_size: Optional[Tuple[int, int]] = None
    process_scale = 1.0

    for position, frame_index in enumerate(
        indexes,
        start=1,
    ):
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index),
        )

        success, frame = capture.read()

        if not success or frame is None:
            continue

        if source_size is None:
            source_size = (
                int(frame.shape[1]),
                int(frame.shape[0]),
            )

        processed, process_scale = resize_for_processing(
            frame,
            PROCESS_MAX_WIDTH,
        )

        # 这里绝对不做 GaussianBlur，保存原始清晰画面。
        (
            sharpness,
            brightness,
            contrast,
            quality_score,
        ) = calculate_quality(processed)

        feature = build_visual_feature(processed)

        raw_candidates.append(
            CandidateFrame(
                frame_index=int(frame_index),
                timestamp_seconds=float(frame_index / fps),
                image=processed,
                feature=feature,
                sharpness=sharpness,
                brightness=brightness,
                contrast=contrast,
                quality_score=quality_score,
            )
        )

        print(
            f"\r正在读取候选帧：{position}/{len(indexes)} "
            f"| 视频帧号：{int(frame_index)}",
            end="",
            flush=True,
        )

    print()

    if source_size is None or not raw_candidates:
        raise RuntimeError("没有成功读取任何有效视频帧。")

    # 根据本视频清晰度分布自动确定阈值。
    sharpness_values = np.array(
        [item.sharpness for item in raw_candidates],
        dtype=np.float32,
    )

    adaptive_threshold = max(
        ABSOLUTE_MIN_SHARPNESS,
        float(np.percentile(sharpness_values, 22)),
    )

    filtered = [
        item
        for item in raw_candidates
        if (
            item.sharpness >= adaptive_threshold
            and MIN_BRIGHTNESS <= item.brightness <= MAX_BRIGHTNESS
            and item.contrast >= MIN_CONTRAST
        )
    ]

    # 如果过滤过多，就退回到清晰度最高的一批帧，避免没有足够候选。
    required_minimum = max(
        REFERENCE_COUNT * 3,
        REFERENCE_COUNT,
    )

    if len(filtered) < required_minimum:
        print(
            "[提示] 严格质量筛选后的帧数不足，"
            "自动使用清晰度和质量最高的一批候选帧。"
        )

        filtered = sorted(
            raw_candidates,
            key=lambda item: (
                item.quality_score,
                item.sharpness,
            ),
            reverse=True,
        )[:max(required_minimum, min(len(raw_candidates), REFERENCE_COUNT * 8))]

    print(
        f"候选帧总数：{len(raw_candidates)} | "
        f"质量筛选后：{len(filtered)} | "
        f"清晰度阈值：{adaptive_threshold:.2f}"
    )

    return filtered, source_size, process_scale


# ============================================================
# 聚类与代表帧选择
# ============================================================

def normalize_features(
    features: np.ndarray,
) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    normalized = (features - mean) / std

    # 限制极端特征，防止个别像素主导聚类。
    normalized = np.clip(
        normalized,
        -4.0,
        4.0,
    )

    return normalized.astype(np.float32)


def cluster_candidates(
    candidates: List[CandidateFrame],
) -> Tuple[List[CandidateFrame], int]:
    cluster_count = min(
        REFERENCE_COUNT,
        len(candidates),
    )

    if cluster_count <= 0:
        raise RuntimeError("没有可用于建立基准库的候选帧。")

    if cluster_count == 1:
        best = max(
            candidates,
            key=lambda item: item.quality_score,
        )
        best.cluster_id = 0
        return [best], 1

    features = np.stack(
        [item.feature for item in candidates],
        axis=0,
    )

    normalized_features = normalize_features(
        features
    )

    cv2.setRNGSeed(RANDOM_SEED)

    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.01,
    )

    _, labels, centers = cv2.kmeans(
        normalized_features,
        cluster_count,
        None,
        criteria,
        12,
        cv2.KMEANS_PP_CENTERS,
    )

    labels = labels.reshape(-1)

    selected: List[CandidateFrame] = []

    all_quality = np.array(
        [item.quality_score for item in candidates],
        dtype=np.float32,
    )

    quality_min = float(all_quality.min())
    quality_range = max(
        float(all_quality.max() - quality_min),
        1e-6,
    )

    for cluster_id in range(cluster_count):
        member_indexes = np.where(
            labels == cluster_id
        )[0]

        if len(member_indexes) == 0:
            continue

        center = centers[cluster_id]

        best_index: Optional[int] = None
        best_score = float("inf")

        for candidate_index in member_indexes:
            candidate = candidates[int(candidate_index)]

            center_distance = float(
                np.mean(
                    np.abs(
                        normalized_features[int(candidate_index)]
                        - center
                    )
                )
            )

            quality_normalized = (
                candidate.quality_score
                - quality_min
            ) / quality_range

            # 优先选择最接近该类中心的帧，同时偏向更清晰的帧。
            score = (
                center_distance
                - 0.32 * quality_normalized
            )

            if score < best_score:
                best_score = score
                best_index = int(candidate_index)

        if best_index is not None:
            chosen = candidates[best_index]
            chosen.cluster_id = int(cluster_id)
            selected.append(chosen)

    selected.sort(
        key=lambda item: item.timestamp_seconds
    )

    return selected, cluster_count


# ============================================================
# 输出
# ============================================================

def make_contact_sheet(
    references: List[CandidateFrame],
    columns: int = 4,
) -> np.ndarray:
    if not references:
        raise RuntimeError("无法生成预览图：基准帧列表为空。")

    thumb_width = 360
    source_h, source_w = references[0].image.shape[:2]
    thumb_height = max(
        1,
        int(round(source_h * thumb_width / source_w)),
    )

    label_height = 42
    tile_height = thumb_height + label_height

    rows = int(math.ceil(len(references) / columns))

    sheet = np.full(
        (
            rows * tile_height,
            columns * thumb_width,
            3,
        ),
        24,
        dtype=np.uint8,
    )

    for index, item in enumerate(references):
        row = index // columns
        column = index % columns

        x1 = column * thumb_width
        y1 = row * tile_height

        thumb = cv2.resize(
            item.image,
            (thumb_width, thumb_height),
            interpolation=cv2.INTER_AREA,
        )

        sheet[
            y1:y1 + thumb_height,
            x1:x1 + thumb_width,
        ] = thumb

        cv2.rectangle(
            sheet,
            (x1, y1 + thumb_height),
            (x1 + thumb_width, y1 + tile_height),
            (12, 18, 26),
            -1,
        )

        label = (
            f"REF {index + 1:02d}  "
            f"t={item.timestamp_seconds:.1f}s  "
            f"sharp={item.sharpness:.0f}"
        )

        cv2.putText(
            sheet,
            label,
            (x1 + 10, y1 + thumb_height + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

    return sheet


def save_outputs(
    project_root: Path,
    references: List[CandidateFrame],
    source_size: Tuple[int, int],
    process_scale: float,
    fps: float,
    total_frames: int,
    duration_seconds: float,
    candidate_count: int,
    actual_cluster_count: int,
) -> None:
    output_root = project_root / OUTPUT_ROOT
    reference_dir = output_root / REFERENCE_DIR_NAME

    # 只清理旧的基准图片库，不删除其他道路异常数据。
    if reference_dir.exists():
        shutil.rmtree(reference_dir)

    reference_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_records = []

    for index, item in enumerate(
        references,
        start=1,
    ):
        filename = f"reference_{index:02d}.png"
        output_path = reference_dir / filename

        imwrite_unicode(
            output_path,
            item.image,
        )

        reference_records.append({
            "reference_id": index,
            "filename": str(
                Path(REFERENCE_DIR_NAME) / filename
            ).replace("\\", "/"),
            "source_frame_index": item.frame_index,
            "timestamp_seconds": round(
                item.timestamp_seconds,
                4,
            ),
            "cluster_id": item.cluster_id,
            "sharpness": round(item.sharpness, 4),
            "brightness": round(item.brightness, 4),
            "contrast": round(item.contrast, 4),
            "quality_score": round(
                item.quality_score,
                6,
            ),
        })

    contact_sheet = make_contact_sheet(
        references
    )

    contact_sheet_path = (
        output_root / "reference_contact_sheet.jpg"
    )

    imwrite_unicode(
        contact_sheet_path,
        contact_sheet,
        [int(cv2.IMWRITE_JPEG_QUALITY), 94],
    )

    processed_height, processed_width = (
        references[0].image.shape[:2]
    )

    metadata = {
        "camera_id": "camera_01",
        "source_video": VIDEO_FILENAME,
        "strategy": "multiple_representative_reference_frames",
        "important": (
            "这些图片彼此独立，没有执行平均值、中位数合成或模糊处理。"
        ),
        "source_width": int(source_size[0]),
        "source_height": int(source_size[1]),
        "source_fps": round(float(fps), 4),
        "source_total_frames": int(total_frames),
        "source_duration_seconds": round(
            float(duration_seconds),
            4,
        ),
        "processed_width": int(processed_width),
        "processed_height": int(processed_height),
        "process_scale": round(
            float(process_scale),
            8,
        ),
        "candidate_count_after_quality_filter": int(
            candidate_count
        ),
        "reference_count_requested": int(
            REFERENCE_COUNT
        ),
        "reference_count_saved": len(references),
        "cluster_count": int(actual_cluster_count),
        "future_runtime_logic": [
            "将实时帧缩放到与基准图相同的处理尺寸。",
            "从多个基准图中选择与当前帧最接近的一张。",
            "对选中的基准图和当前帧执行小范围图像配准。",
            "使用道路 ROI 限制检测范围。",
            "使用 YOLO 车辆 Mask 排除车辆区域。",
            "对剩余区域计算局部异常差异。",
            "异常区域连续存在达到阈值后才报警。",
        ],
        "references": reference_records,
    }

    json_path = (
        output_root / "reference_bank.json"
    )

    json_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    print("多基准道路图片库生成完成")
    print(f"基准图片目录：{reference_dir}")
    print(f"基准图片数量：{len(references)}")
    print(f"总览预览图片：{contact_sheet_path}")
    print(f"基准索引配置：{json_path}")
    print("=" * 76)
    print(
        "请打开 reference_contact_sheet.jpg，"
        "确认每一张都是清晰的正常道路画面。"
    )


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    project_root = Path(__file__).resolve().parent
    video_path = project_root / VIDEO_FILENAME

    print("=" * 76)
    print("正常道路多基准图片库生成")
    print(f"项目目录：{project_root}")
    print(f"输入视频：{video_path}")
    print(f"计划保存基准数量：{REFERENCE_COUNT}")
    print("=" * 76)

    if not video_path.exists():
        raise FileNotFoundError(
            f"未找到视频：{video_path}\n"
            f"请确认视频位于项目根目录，文件名严格为：{VIDEO_FILENAME}"
        )

    capture, temp_video_path = open_video_with_fallback(
        video_path
    )

    try:
        fps = float(
            capture.get(cv2.CAP_PROP_FPS)
        )

        total_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if (
            not np.isfinite(fps)
            or fps <= 0
        ):
            fps = 25.0

        duration_seconds = (
            total_frames / fps
            if total_frames > 0
            else 0.0
        )

        print(f"视频 FPS：{fps:.2f}")
        print(f"视频总帧数：{total_frames}")
        print(f"视频时长：{duration_seconds:.2f} 秒")

        sample_indexes = build_sample_indexes(
            total_frames,
            fps,
        )

        (
            candidates,
            source_size,
            process_scale,
        ) = read_candidates(
            capture,
            sample_indexes,
            fps,
        )
    finally:
        capture.release()

        if temp_video_path is not None:
            temp_video_path.unlink(
                missing_ok=True
            )

    print("正在对不同晃动位置和光照状态进行聚类……")

    references, actual_cluster_count = (
        cluster_candidates(candidates)
    )

    save_outputs(
        project_root=project_root,
        references=references,
        source_size=source_size,
        process_scale=process_scale,
        fps=fps,
        total_frames=total_frames,
        duration_seconds=duration_seconds,
        candidate_count=len(candidates),
        actual_cluster_count=actual_cluster_count,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print(
            f"[失败] {type(exc).__name__}: {exc}"
        )
        sys.exit(1)

# -*- coding: utf-8 -*-
"""
多基准道路障碍物检测 · GPU V4 核心逻辑恢复版

本脚本建立在前一步 road_roi_vehicle_mask_demo.py 之上。

本版原则：
- 恢复准确版核心逻辑：每张候选基准图都与当前帧直接 ORB 匹配，
  直接计算该基准图到当前帧的 Homography；禁止使用离线矩阵组合推算。
- 保留旧版 LAB + CLAHE + 纹理 + Canny + 中值滤波差分算法。
- normal.onnx 强制使用 CUDAExecutionProvider。
- 只做不会改变检测语义的性能优化：7张直接配准并行、3张精确差分并行、
  只对最终选中的3张基准做 warpPerspective。

完整流程：
1. 读取视频帧。
2. 道路 ROI 跟随摄像头晃动。
3. normal.onnx 检测当前车辆，生成当前车辆忽略 Mask。
4. 从多个正常道路基准图中，筛选画面最接近的候选。
5. 使用 ORB + Homography，把每张候选基准图对齐到当前帧。
6. 同时排除：
   - 当前帧车辆
   - 基准图中的车辆
   - 道路边界附近不稳定区域
7. 分别计算灰度、颜色、纹理和边缘差异。
8. 三张基准图至少两张同时认为异常，才保留该区域。
9. 异常区域必须连续存在，才从“候选”升级为“确认障碍物”。

默认先测试：
    正常道路.mp4

也可以指定其他视频：
    python multi_reference_road_anomaly_demo.py 异常道路.mp4

输出：
road_anomaly_data/camera_01/
    road_anomaly_result.mp4
    reference_vehicle_masks/
    reference_vehicle_previews/
    reference_runtime_cache.json

依赖：
- normal.onnx
- road_roi_vehicle_mask_demo.py
- 已生成的 reference_bank
- 已生成的 road_masks
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


GPU_PIPELINE_VERSION = "2026-07-14-GPU-V4-DIRECT-ALIGNMENT"
print("=" * 80)
print(f"[启动版本] {GPU_PIPELINE_VERSION}")
print(f"[启动文件] {Path(__file__).resolve()}")
print("=" * 80)

# Windows 下先导入 PyTorch，让 PyTorch 加载与自身匹配的 CUDA/cuDNN DLL；
# 随后 ONNX Runtime 复用同一套 DLL，避免 WinError 127。
try:
    import torch
except Exception as exc:
    raise RuntimeError(
        "PyTorch CUDA 导入失败。请确认 GPU V3 环境检测已经通过。"
        f"\n原始错误：{exc}"
    ) from exc

if not torch.cuda.is_available():
    raise RuntimeError(
        "PyTorch 已导入，但 torch.cuda.is_available() 为 False。"
        f"\nPyTorch：{torch.__version__}"
        f"\nPyTorch CUDA：{getattr(torch.version, 'cuda', None)}"
    )

try:
    _cuda_probe = torch.zeros((1,), device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    del _cuda_probe
except Exception as exc:
    raise RuntimeError(f"PyTorch CUDA 初始化失败：{exc}") from exc

print(
    "[CUDA] PyTorch 已加载 CUDA/cuDNN："
    f"torch={torch.__version__} | CUDA={torch.version.cuda} | "
    f"GPU={torch.cuda.get_device_name(0)}"
)

try:
    import onnxruntime as ort
except Exception as exc:
    raise RuntimeError(f"onnxruntime-gpu 导入失败：{exc}") from exc

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls(
        cuda=True,
        cudnn=True,
        msvc=True,
        directory=None,
    )

_AVAILABLE_PROVIDERS = list(ort.get_available_providers())
print(f"[CUDA] ONNX Runtime Provider：{_AVAILABLE_PROVIDERS}")

if "CUDAExecutionProvider" not in _AVAILABLE_PROVIDERS:
    raise RuntimeError(
        "当前环境没有 CUDAExecutionProvider："
        f"{_AVAILABLE_PROVIDERS}"
    )

try:
    from road_roi_vehicle_mask_demo import (
        NormalOnnxVehicleDetector,
        RoadRoiTracker,
        build_vehicle_mask,
        imread_unicode,
        load_roi_points,
        open_video_with_fallback,
        polygon_to_mask,
        resize_to_fit,
    )
except Exception as exc:
    raise RuntimeError(
        "无法导入 road_roi_vehicle_mask_demo.py。\n"
        "请确认本脚本和 road_roi_vehicle_mask_demo.py "
        f"都位于项目根目录。\n原始错误：{exc}"
    ) from exc


# ============================================================
# 路径和输入
# ============================================================

DEFAULT_VIDEO_FILENAME = "正常道路.mp4"
MODEL_FILENAME = "normal.onnx"

CAMERA_ROOT = Path("road_anomaly_data") / "camera_01"
REFERENCE_DIR = CAMERA_ROOT / "reference_bank"
ROAD_MASK_DIR = CAMERA_ROOT / "road_masks"
ANCHOR_IMAGE_PATH = REFERENCE_DIR / "reference_01.png"
ANCHOR_ROI_JSON_PATH = CAMERA_ROOT / "road_roi" / "road_roi.json"
ANCHOR_ROAD_MASK_PATH = CAMERA_ROOT / "road_roi" / "road_mask.png"

REFERENCE_VEHICLE_MASK_DIR = (
    CAMERA_ROOT / "reference_vehicle_masks"
)
REFERENCE_VEHICLE_PREVIEW_DIR = (
    CAMERA_ROOT / "reference_vehicle_previews"
)
REFERENCE_CACHE_JSON = (
    CAMERA_ROOT / "reference_runtime_cache.json"
)

OUTPUT_VIDEO_PATH = (
    CAMERA_ROOT / "road_anomaly_result.mp4"
)


# ============================================================
# 处理参数
# ============================================================

WINDOW_NAME = "Multi-reference Road Anomaly Detection - GPU V4 Direct Alignment"
SAVE_OUTPUT_VIDEO = True

MAX_DISPLAY_WIDTH = 1750
MAX_DISPLAY_HEIGHT = 950

# normal.onnx 每帧执行。
YOLO_EVERY_N_FRAMES = 1

# 道路障碍物多基准比较每隔几帧执行一次。
# 3 表示约每3帧检测一次，其余帧复用最近结果。
ANOMALY_EVERY_N_FRAMES = 3

# 先使用轻量特征筛选最接近的几张，再做 ORB 几何匹配。
REFERENCE_SHORTLIST_COUNT = 7

# 最终参与投票的基准图数量。
TOP_REFERENCE_COUNT = 3

# 至少多少张基准图同时认为该像素异常。
MIN_REFERENCE_VOTES = 2

# ORB 匹配参数。
ORB_FEATURES = 5000
ORB_LOWE_RATIO = 0.76
MIN_GOOD_MATCHES = 22
MIN_INLIERS = 14
MIN_INLIER_RATIO = 0.25

# 对齐结果限制。
MIN_WARP_AREA_RATIO = 0.70
MAX_WARP_AREA_RATIO = 1.35
MAX_CENTER_SHIFT_RATIO = 0.27
MAX_PERSPECTIVE_TERM = 0.005

# 道路和车辆边界附近容易因对齐误差产生变化，向内收缩。
ROAD_BOUNDARY_ERODE = 5

# 基准/当前车辆区域额外膨胀，避免车辆边缘和阴影产生差分。
VEHICLE_EXTRA_DILATE = 7

# 差异阈值。
# 实际每张基准还会根据有效区域噪声动态调整。
BASE_DIFF_THRESHOLD = 37.0
MAX_ADAPTIVE_THRESHOLD = 72.0
MAD_MULTIPLIER = 4.2
MAD_EXTRA = 7.0

# 二值差分形态学参数。
OPEN_KERNEL_SIZE = 3
CLOSE_KERNEL_SIZE = 9

# 连通区域面积。
# 近处使用较大的面积门槛，远处根据透视自动降低。
BOTTOM_MIN_AREA = 170
TOP_MIN_AREA = 34

# 最大异常区域，避免整幅光照变化被当作一个障碍物。
MAX_COMPONENT_ROAD_RATIO = 0.16

# 连通区域平均差异分数要求。
MIN_COMPONENT_MEAN_SCORE = 42.0

# 时间确认。
CONFIRM_MIN_SECONDS = 1.25
CONFIRM_MIN_HITS = 5
MAX_TRACK_MISSES = 8

# 异常候选框匹配。
TRACK_IOU_THRESHOLD = 0.10
TRACK_CENTER_DISTANCE_RATIO = 1.25

# 车辆掩膜参考图缓存是否强制重建。
REBUILD_REFERENCE_VEHICLE_MASKS = False

# 不改变算法语义的并行优化。
# 每张基准仍然直接与当前帧做 ORB/Homography，只是并行执行。
ALIGNMENT_WORKERS = max(1, min(4, int(os.cpu_count() or 1)))
DIFFERENCE_WORKERS = max(1, min(3, int(os.cpu_count() or 1)))


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ReferenceEntry:
    name: str
    image_path: Path
    image: np.ndarray
    road_mask: np.ndarray
    vehicle_mask: np.ndarray
    scene_feature: np.ndarray
    gray_feature: np.ndarray
    keypoints: Sequence[Any]
    descriptors: np.ndarray


@dataclass
class AlignedReference:
    entry: ReferenceEntry
    homography: np.ndarray
    good_matches: int
    inliers: int
    inlier_ratio: float
    warped_image: np.ndarray
    warped_road_mask: np.ndarray
    warped_vehicle_mask: np.ndarray


@dataclass
class CandidateRegion:
    bbox: Tuple[int, int, int, int]
    area: int
    mean_score: float
    vote_mean: float


@dataclass
class AnomalyTrack:
    track_id: int
    bbox: Tuple[int, int, int, int]
    first_seen: float
    last_seen: float
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    mean_score: float = 0.0
    vote_mean: float = 0.0
    matched_this_update: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)


# ============================================================
# 写文件辅助
# ============================================================

def imwrite_unicode(
    path: Path,
    image: np.ndarray,
    params: Optional[List[int]] = None,
) -> None:
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


# ============================================================
# 场景轻量特征
# ============================================================

def build_scene_feature(
    image: np.ndarray,
) -> np.ndarray:
    """
    用低分辨率结构 + 亮度直方图做快速初筛。
    这里只用于从16张基准里挑出若干候选，不作为异常判断。
    """
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    small = cv2.resize(
        gray,
        (32, 18),
        interpolation=cv2.INTER_AREA,
    )

    equalized = cv2.equalizeHist(small)

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
        edges.reshape(-1).astype(np.float32) / 255.0 * 0.75,
        histogram.astype(np.float32) * 4.0,
    ])

    return feature.astype(np.float32)


# ============================================================
# ORB 基准管理
# ============================================================

class ReferenceBank:
    def __init__(
        self,
        project_root: Path,
        detector: NormalOnnxVehicleDetector,
        anchor_size: Tuple[int, int],
    ):
        self.project_root = project_root
        self.detector = detector
        self.width, self.height = anchor_size

        self.orb = cv2.ORB_create(
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

        self.matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=False,
        )

        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        self.entries: List[ReferenceEntry] = []
        self._load_all()

    def normalize_gray(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
        return self.clahe.apply(gray)

    def _load_road_mask(
        self,
        reference_path: Path,
    ) -> Optional[np.ndarray]:
        mapped_mask_path = (
            self.project_root
            / ROAD_MASK_DIR
            / f"{reference_path.stem}_road_mask.png"
        )

        if mapped_mask_path.exists():
            mask = imread_unicode(
                mapped_mask_path,
                cv2.IMREAD_GRAYSCALE,
            )

            if mask is not None:
                return mask

        if reference_path.name == "reference_01.png":
            anchor_mask_path = (
                self.project_root
                / ANCHOR_ROAD_MASK_PATH
            )

            if anchor_mask_path.exists():
                return imread_unicode(
                    anchor_mask_path,
                    cv2.IMREAD_GRAYSCALE,
                )

        return None

    def _reference_vehicle_paths(
        self,
        reference_path: Path,
    ) -> Tuple[Path, Path]:
        mask_path = (
            self.project_root
            / REFERENCE_VEHICLE_MASK_DIR
            / f"{reference_path.stem}_vehicle_mask.png"
        )

        preview_path = (
            self.project_root
            / REFERENCE_VEHICLE_PREVIEW_DIR
            / f"{reference_path.stem}_vehicle_preview.jpg"
        )

        return mask_path, preview_path

    def _create_reference_vehicle_mask(
        self,
        image: np.ndarray,
        road_mask: np.ndarray,
        reference_path: Path,
    ) -> np.ndarray:
        mask_path, preview_path = (
            self._reference_vehicle_paths(
                reference_path
            )
        )

        if (
            mask_path.exists()
            and not REBUILD_REFERENCE_VEHICLE_MASKS
        ):
            cached = imread_unicode(
                mask_path,
                cv2.IMREAD_GRAYSCALE,
            )

            if (
                cached is not None
                and cached.shape[:2]
                == road_mask.shape[:2]
            ):
                _, cached = cv2.threshold(
                    cached,
                    127,
                    255,
                    cv2.THRESH_BINARY,
                )
                return cached

        detections, inference_ms = (
            self.detector.detect(image)
        )

        vehicle_mask, accepted = build_vehicle_mask(
            detections,
            self.detector,
            road_mask,
        )

        if VEHICLE_EXTRA_DILATE > 0:
            kernel = np.ones(
                (
                    VEHICLE_EXTRA_DILATE,
                    VEHICLE_EXTRA_DILATE,
                ),
                dtype=np.uint8,
            )

            vehicle_mask = cv2.dilate(
                vehicle_mask,
                kernel,
                iterations=1,
            )

            vehicle_mask = cv2.bitwise_and(
                vehicle_mask,
                road_mask,
            )

        imwrite_unicode(
            mask_path,
            vehicle_mask,
        )

        preview = image.copy()

        layer = preview.copy()
        layer[road_mask > 0] = (50, 170, 95)
        preview = cv2.addWeighted(
            layer,
            0.13,
            preview,
            0.87,
            0,
        )

        vehicle_layer = preview.copy()
        vehicle_layer[vehicle_mask > 0] = (
            30,
            30,
            235,
        )
        preview = cv2.addWeighted(
            vehicle_layer,
            0.35,
            preview,
            0.65,
            0,
        )

        cv2.rectangle(
            preview,
            (0, 0),
            (preview.shape[1], 55),
            (9, 16, 25),
            -1,
        )

        cv2.putText(
            preview,
            (
                f"{reference_path.name} | "
                f"vehicles={len(accepted)} | "
                f"infer={inference_ms:.1f}ms"
            ),
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )

        imwrite_unicode(
            preview_path,
            preview,
            [int(cv2.IMWRITE_JPEG_QUALITY), 94],
        )

        return vehicle_mask

    def _build_orb_mask(
        self,
        road_mask: np.ndarray,
        vehicle_mask: np.ndarray,
    ) -> np.ndarray:
        # ORB 匹配时允许使用整张画面的固定设施，
        # 但屏蔽车辆及其附近，避免移动汽车参与几何估计。
        vehicle_expanded = cv2.dilate(
            vehicle_mask,
            np.ones((17, 17), np.uint8),
            iterations=1,
        )

        return cv2.bitwise_not(
            vehicle_expanded
        )

    def _load_all(self) -> None:
        reference_dir = (
            self.project_root / REFERENCE_DIR
        )

        reference_paths = sorted(
            reference_dir.glob(
                "reference_*.png"
            )
        )

        if not reference_paths:
            raise FileNotFoundError(
                f"基准目录中没有 reference_*.png："
                f"{reference_dir}"
            )

        cache_records = []

        print("=" * 80)
        print("正在准备正常道路多基准库")
        print(f"基准图数量：{len(reference_paths)}")
        print("=" * 80)

        for index, reference_path in enumerate(
            reference_paths,
            start=1,
        ):
            image = imread_unicode(
                reference_path,
                cv2.IMREAD_COLOR,
            )

            if image is None:
                print(
                    f"[跳过] 无法读取："
                    f"{reference_path.name}"
                )
                continue

            image = cv2.resize(
                image,
                (self.width, self.height),
                interpolation=cv2.INTER_AREA,
            )

            road_mask = self._load_road_mask(
                reference_path
            )

            if road_mask is None:
                print(
                    f"[跳过] 缺少道路 Mask："
                    f"{reference_path.name}"
                )
                continue

            road_mask = cv2.resize(
                road_mask,
                (self.width, self.height),
                interpolation=cv2.INTER_NEAREST,
            )

            _, road_mask = cv2.threshold(
                road_mask,
                127,
                255,
                cv2.THRESH_BINARY,
            )

            vehicle_mask = (
                self._create_reference_vehicle_mask(
                    image,
                    road_mask,
                    reference_path,
                )
            )

            orb_mask = self._build_orb_mask(
                road_mask,
                vehicle_mask,
            )

            normalized_gray = self.normalize_gray(
                image
            )

            keypoints, descriptors = (
                self.orb.detectAndCompute(
                    normalized_gray,
                    orb_mask,
                )
            )

            if (
                descriptors is None
                or len(keypoints) < MIN_GOOD_MATCHES
            ):
                print(
                    f"[跳过] ORB 特征不足："
                    f"{reference_path.name}"
                )
                continue

            entry = ReferenceEntry(
                name=reference_path.name,
                image_path=reference_path,
                image=image,
                road_mask=road_mask,
                vehicle_mask=vehicle_mask,
                scene_feature=build_scene_feature(
                    image
                ),
                gray_feature=normalized_gray,
                keypoints=keypoints,
                descriptors=descriptors,
            )

            self.entries.append(entry)

            cache_records.append({
                "reference": reference_path.name,
                "keypoint_count": len(keypoints),
                "road_pixels": int(
                    np.count_nonzero(
                        road_mask
                    )
                ),
                "vehicle_pixels": int(
                    np.count_nonzero(
                        vehicle_mask
                    )
                ),
            })

            print(
                f"[{index}/{len(reference_paths)}] "
                f"{reference_path.name}："
                f"ORB={len(keypoints)}"
            )

        if len(self.entries) < TOP_REFERENCE_COUNT:
            raise RuntimeError(
                "可用基准图不足。至少需要"
                f"{TOP_REFERENCE_COUNT}张，"
                f"当前只有{len(self.entries)}张。"
            )

        cache_path = (
            self.project_root
            / REFERENCE_CACHE_JSON
        )

        cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cache_path.write_text(
            json.dumps(
                {
                    "reference_count": len(
                        self.entries
                    ),
                    "records": cache_records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print("=" * 80)
        print(
            f"基准库准备完成："
            f"{len(self.entries)}张可用"
        )
        print("=" * 80)

    def shortlist(
        self,
        current_frame: np.ndarray,
    ) -> List[ReferenceEntry]:
        current_feature = build_scene_feature(
            current_frame
        )

        scored = []

        for entry in self.entries:
            distance = float(
                np.mean(
                    np.abs(
                        current_feature
                        - entry.scene_feature
                    )
                )
            )

            scored.append(
                (distance, entry)
            )

        scored.sort(
            key=lambda item: item[0]
        )

        return [
            entry
            for _, entry in scored[
                :min(
                    REFERENCE_SHORTLIST_COUNT,
                    len(scored),
                )
            ]
        ]

    def _validate_homography(
        self,
        homography: np.ndarray,
    ) -> bool:
        corners = np.asarray(
            [
                [0.0, 0.0],
                [self.width - 1.0, 0.0],
                [
                    self.width - 1.0,
                    self.height - 1.0,
                ],
                [0.0, self.height - 1.0],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        transformed = cv2.perspectiveTransform(
            corners,
            homography,
        ).reshape(-1, 2)

        transformed_contour = (
            np.round(transformed)
            .astype(np.int32)
            .reshape((-1, 1, 2))
        )

        if not cv2.isContourConvex(
            transformed_contour
        ):
            return False

        original_area = float(
            self.width * self.height
        )

        transformed_area = abs(
            float(
                cv2.contourArea(
                    transformed.astype(
                        np.float32
                    )
                )
            )
        )

        area_ratio = transformed_area / max(
            original_area,
            1.0,
        )

        source_center = np.asarray(
            [
                self.width / 2.0,
                self.height / 2.0,
            ],
            dtype=np.float32,
        )

        target_center = transformed.mean(
            axis=0
        )

        center_shift = float(
            np.linalg.norm(
                target_center
                - source_center
            )
        )

        max_center_shift = (
            math.hypot(
                self.width,
                self.height,
            )
            * MAX_CENTER_SHIFT_RATIO
        )

        perspective_term = max(
            abs(float(homography[2, 0])),
            abs(float(homography[2, 1])),
        )

        return bool(
            MIN_WARP_AREA_RATIO
            <= area_ratio
            <= MAX_WARP_AREA_RATIO
            and center_shift
            <= max_center_shift
            and perspective_term
            <= MAX_PERSPECTIVE_TERM
        )

    def _match_one_reference(
        self,
        entry: ReferenceEntry,
        current_keypoints: Sequence[Any],
        current_descriptors: np.ndarray,
    ) -> Optional[AlignedReference]:
        """每张基准图直接与当前帧匹配，不使用任何离线矩阵组合。"""
        matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=False,
        )

        knn_matches = matcher.knnMatch(
            entry.descriptors,
            current_descriptors,
            k=2,
        )

        good_matches = []
        for pair in knn_matches:
            if len(pair) != 2:
                continue
            first, second = pair
            if first.distance < ORB_LOWE_RATIO * second.distance:
                good_matches.append(first)

        if len(good_matches) < MIN_GOOD_MATCHES:
            return None

        reference_points = np.float32(
            [
                entry.keypoints[match.queryIdx].pt
                for match in good_matches
            ]
        ).reshape(-1, 1, 2)

        current_points = np.float32(
            [
                current_keypoints[match.trainIdx].pt
                for match in good_matches
            ]
        ).reshape(-1, 1, 2)

        homography, inlier_mask = cv2.findHomography(
            reference_points,
            current_points,
            cv2.RANSAC,
            3.5,
            maxIters=3500,
            confidence=0.995,
        )

        if homography is None or inlier_mask is None:
            return None

        inliers = int(inlier_mask.reshape(-1).sum())
        inlier_ratio = inliers / max(len(good_matches), 1)

        if (
            inliers < MIN_INLIERS
            or inlier_ratio < MIN_INLIER_RATIO
            or not self._validate_homography(homography)
        ):
            return None

        # 先不执行图像变换。等所有候选按几何质量排序后，
        # 只对最终最好的3张做 warpPerspective，减少无意义的变换。
        return AlignedReference(
            entry=entry,
            homography=homography,
            good_matches=len(good_matches),
            inliers=inliers,
            inlier_ratio=float(inlier_ratio),
            warped_image=None,
            warped_road_mask=None,
            warped_vehicle_mask=None,
        )

    def _warp_selected_reference(
        self,
        aligned: AlignedReference,
    ) -> AlignedReference:
        aligned.warped_image = cv2.warpPerspective(
            aligned.entry.image,
            aligned.homography,
            (self.width, self.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        aligned.warped_road_mask = cv2.warpPerspective(
            aligned.entry.road_mask,
            aligned.homography,
            (self.width, self.height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        aligned.warped_vehicle_mask = cv2.warpPerspective(
            aligned.entry.vehicle_mask,
            aligned.homography,
            (self.width, self.height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        _, aligned.warped_road_mask = cv2.threshold(
            aligned.warped_road_mask,
            127,
            255,
            cv2.THRESH_BINARY,
        )
        _, aligned.warped_vehicle_mask = cv2.threshold(
            aligned.warped_vehicle_mask,
            127,
            255,
            cv2.THRESH_BINARY,
        )
        return aligned

    def align_best(
        self,
        current_frame: np.ndarray,
        current_vehicle_mask: np.ndarray,
    ) -> List[AlignedReference]:
        current_gray = self.normalize_gray(current_frame)

        current_orb_mask = cv2.bitwise_not(
            cv2.dilate(
                current_vehicle_mask,
                np.ones((17, 17), dtype=np.uint8),
                iterations=1,
            )
        )

        current_keypoints, current_descriptors = self.orb.detectAndCompute(
            current_gray,
            current_orb_mask,
        )

        if (
            current_descriptors is None
            or len(current_keypoints) < MIN_GOOD_MATCHES
        ):
            return []

        shortlist = self.shortlist(current_frame)
        matched: List[AlignedReference] = []

        # 逻辑与旧版相同：7张候选逐张直接匹配当前帧。
        # 这里只把互不依赖的7次匹配并行执行。
        with ThreadPoolExecutor(max_workers=ALIGNMENT_WORKERS) as executor:
            futures = [
                executor.submit(
                    self._match_one_reference,
                    entry,
                    current_keypoints,
                    current_descriptors,
                )
                for entry in shortlist
            ]

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    matched.append(result)

        matched.sort(
            key=lambda item: (item.inliers, item.inlier_ratio),
            reverse=True,
        )
        selected = matched[: min(TOP_REFERENCE_COUNT, len(matched))]

        # 只对真正选中的3张执行图像和Mask变换。
        if len(selected) > 1:
            with ThreadPoolExecutor(
                max_workers=min(len(selected), DIFFERENCE_WORKERS)
            ) as executor:
                selected = list(
                    executor.map(self._warp_selected_reference, selected)
                )
        elif selected:
            selected = [self._warp_selected_reference(selected[0])]

        return selected


# ============================================================
# 图像差异
# ============================================================

def create_effective_mask(
    current_road_mask: np.ndarray,
    current_vehicle_mask: np.ndarray,
    aligned: AlignedReference,
) -> np.ndarray:
    reference_vehicle = (
        aligned.warped_vehicle_mask
    )

    if VEHICLE_EXTRA_DILATE > 0:
        kernel = np.ones(
            (
                VEHICLE_EXTRA_DILATE,
                VEHICLE_EXTRA_DILATE,
            ),
            dtype=np.uint8,
        )

        current_vehicle = cv2.dilate(
            current_vehicle_mask,
            kernel,
            iterations=1,
        )

        reference_vehicle = cv2.dilate(
            reference_vehicle,
            kernel,
            iterations=1,
        )
    else:
        current_vehicle = (
            current_vehicle_mask
        )

    valid = cv2.bitwise_and(
        current_road_mask,
        aligned.warped_road_mask,
    )

    invalid_vehicles = cv2.bitwise_or(
        current_vehicle,
        reference_vehicle,
    )

    valid = cv2.bitwise_and(
        valid,
        cv2.bitwise_not(
            invalid_vehicles
        ),
    )

    if ROAD_BOUNDARY_ERODE > 0:
        kernel = np.ones(
            (
                ROAD_BOUNDARY_ERODE,
                ROAD_BOUNDARY_ERODE,
            ),
            dtype=np.uint8,
        )

        valid = cv2.erode(
            valid,
            kernel,
            iterations=1,
        )

    return valid


def normalize_luminance(
    current_l: np.ndarray,
    reference_l: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用有效道路区域的均值和标准差做全局亮度归一化，
    降低曝光轻微变化带来的整片误报。
    """
    valid = valid_mask > 0

    if np.count_nonzero(valid) < 100:
        return current_l, reference_l

    current_values = (
        current_l[valid].astype(
            np.float32
        )
    )

    reference_values = (
        reference_l[valid].astype(
            np.float32
        )
    )

    current_mean = float(
        current_values.mean()
    )

    reference_mean = float(
        reference_values.mean()
    )

    current_std = max(
        float(current_values.std()),
        6.0,
    )

    reference_std = max(
        float(reference_values.std()),
        6.0,
    )

    adjusted_reference = (
        (
            reference_l.astype(
                np.float32
            )
            - reference_mean
        )
        * (
            current_std
            / reference_std
        )
        + current_mean
    )

    adjusted_reference = np.clip(
        adjusted_reference,
        0,
        255,
    ).astype(np.uint8)

    return current_l, adjusted_reference


def compute_difference_score(
    current_frame: np.ndarray,
    reference_frame: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, float]:
    current_lab = cv2.cvtColor(
        current_frame,
        cv2.COLOR_BGR2LAB,
    )

    reference_lab = cv2.cvtColor(
        reference_frame,
        cv2.COLOR_BGR2LAB,
    )

    current_l, current_a, current_b = (
        cv2.split(current_lab)
    )

    reference_l, reference_a, reference_b = (
        cv2.split(reference_lab)
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    current_l = clahe.apply(
        current_l
    )

    reference_l = clahe.apply(
        reference_l
    )

    current_l, reference_l = (
        normalize_luminance(
            current_l,
            reference_l,
            valid_mask,
        )
    )

    luminance_diff = cv2.absdiff(
        current_l,
        reference_l,
    ).astype(np.float32)

    color_a_diff = cv2.absdiff(
        current_a,
        reference_a,
    ).astype(np.float32)

    color_b_diff = cv2.absdiff(
        current_b,
        reference_b,
    ).astype(np.float32)

    color_diff = (
        color_a_diff
        + color_b_diff
    ) * 0.5

    # 高频纹理变化。
    current_detail = cv2.absdiff(
        current_l,
        cv2.GaussianBlur(
            current_l,
            (0, 0),
            5.0,
        ),
    ).astype(np.float32)

    reference_detail = cv2.absdiff(
        reference_l,
        cv2.GaussianBlur(
            reference_l,
            (0, 0),
            5.0,
        ),
    ).astype(np.float32)

    texture_diff = cv2.absdiff(
        current_detail.astype(np.uint8),
        reference_detail.astype(np.uint8),
    ).astype(np.float32)

    current_edges = cv2.Canny(
        current_l,
        60,
        145,
    )

    reference_edges = cv2.Canny(
        reference_l,
        60,
        145,
    )

    edge_diff = cv2.absdiff(
        current_edges,
        reference_edges,
    ).astype(np.float32)

    score = (
        0.50 * luminance_diff
        + 0.20 * color_diff
        + 0.18 * texture_diff
        + 0.12 * edge_diff
    )

    # 这里只平滑“差异分数”，不会把基准图或原视频变模糊。
    score = cv2.medianBlur(
        np.clip(
            score,
            0,
            255,
        ).astype(np.uint8),
        5,
    ).astype(np.float32)

    score[valid_mask == 0] = 0.0

    valid_values = score[
        valid_mask > 0
    ]

    if valid_values.size == 0:
        return score, BASE_DIFF_THRESHOLD

    median = float(
        np.median(valid_values)
    )

    mad = float(
        np.median(
            np.abs(
                valid_values
                - median
            )
        )
    )

    adaptive_threshold = max(
        BASE_DIFF_THRESHOLD,
        median
        + MAD_MULTIPLIER * mad
        + MAD_EXTRA,
    )

    adaptive_threshold = min(
        adaptive_threshold,
        MAX_ADAPTIVE_THRESHOLD,
    )

    return score, float(
        adaptive_threshold
    )


def perspective_min_area(
    center_y: float,
    image_height: int,
) -> int:
    ratio = np.clip(
        center_y / max(
            image_height - 1,
            1,
        ),
        0.0,
        1.0,
    )

    return int(
        round(
            TOP_MIN_AREA
            + (
                BOTTOM_MIN_AREA
                - TOP_MIN_AREA
            )
            * ratio * ratio
        )
    )


def extract_candidate_regions(
    binary_mask: np.ndarray,
    combined_score: np.ndarray,
    vote_map: np.ndarray,
    road_mask: np.ndarray,
) -> List[CandidateRegion]:
    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8,
        )
    )

    road_pixels = max(
        int(
            np.count_nonzero(
                road_mask
            )
        ),
        1,
    )

    candidates = []

    for label_id in range(1, count):
        x = int(
            stats[
                label_id,
                cv2.CC_STAT_LEFT,
            ]
        )

        y = int(
            stats[
                label_id,
                cv2.CC_STAT_TOP,
            ]
        )

        width = int(
            stats[
                label_id,
                cv2.CC_STAT_WIDTH,
            ]
        )

        height = int(
            stats[
                label_id,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        area = int(
            stats[
                label_id,
                cv2.CC_STAT_AREA,
            ]
        )

        center_y = float(
            centroids[label_id][1]
        )

        minimum_area = (
            perspective_min_area(
                center_y,
                binary_mask.shape[0],
            )
        )

        if area < minimum_area:
            continue

        if (
            area / float(road_pixels)
            > MAX_COMPONENT_ROAD_RATIO
        ):
            continue

        if width < 6 or height < 6:
            continue

        component_mask = (
            labels == label_id
        )

        mean_score = float(
            combined_score[
                component_mask
            ].mean()
        )

        vote_mean = float(
            vote_map[
                component_mask
            ].mean()
        )

        if (
            mean_score
            < MIN_COMPONENT_MEAN_SCORE
        ):
            continue

        candidates.append(
            CandidateRegion(
                bbox=(
                    x,
                    y,
                    x + width,
                    y + height,
                ),
                area=area,
                mean_score=mean_score,
                vote_mean=vote_mean,
            )
        )

    return candidates


def _compare_one_aligned_reference(
    current_frame: np.ndarray,
    current_road_mask: np.ndarray,
    current_vehicle_mask: np.ndarray,
    aligned: AlignedReference,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """完全复用旧版有效Mask与差异算法。"""
    valid_mask = create_effective_mask(
        current_road_mask,
        current_vehicle_mask,
        aligned,
    )

    score, threshold = compute_difference_score(
        current_frame,
        aligned.warped_image,
        valid_mask,
    )

    binary = (
        (score >= threshold)
        & (valid_mask > 0)
    ).astype(np.uint8) * 255

    return binary, score, float(threshold), valid_mask


def run_multi_reference_difference(
    current_frame: np.ndarray,
    current_road_mask: np.ndarray,
    current_vehicle_mask: np.ndarray,
    aligned_references: List[AlignedReference],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[CandidateRegion],
    List[float],
]:
    if len(aligned_references) < MIN_REFERENCE_VOTES:
        empty = np.zeros_like(current_road_mask)
        return empty, empty.astype(np.float32), [], []

    # 每张基准的算法与旧版完全一致，只将3个独立比较并行执行。
    with ThreadPoolExecutor(
        max_workers=min(DIFFERENCE_WORKERS, len(aligned_references))
    ) as executor:
        results = list(
            executor.map(
                lambda aligned: _compare_one_aligned_reference(
                    current_frame,
                    current_road_mask,
                    current_vehicle_mask,
                    aligned,
                ),
                aligned_references,
            )
        )

    vote_layers = []
    score_layers = []
    thresholds: List[float] = []
    valid_union = np.zeros_like(current_road_mask)

    for binary, score, threshold, valid_mask in results:
        valid_union = cv2.bitwise_or(valid_union, valid_mask)
        vote_layers.append(binary > 0)
        score_layers.append(score)
        thresholds.append(threshold)

    vote_map = np.sum(
        np.stack(vote_layers, axis=0),
        axis=0,
    ).astype(np.uint8)

    combined_score = np.mean(
        np.stack(score_layers, axis=0),
        axis=0,
    ).astype(np.float32)

    voted_binary = (
        (vote_map >= MIN_REFERENCE_VOTES)
        & (valid_union > 0)
    ).astype(np.uint8) * 255

    open_kernel = np.ones(
        (OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE),
        dtype=np.uint8,
    )
    close_kernel = np.ones(
        (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE),
        dtype=np.uint8,
    )

    # 保留旧版 OpenCV 形态学，避免GPU池化边界语义差异。
    voted_binary = cv2.morphologyEx(
        voted_binary,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1,
    )
    voted_binary = cv2.morphologyEx(
        voted_binary,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )

    candidates = extract_candidate_regions(
        voted_binary,
        combined_score,
        vote_map.astype(np.float32),
        current_road_mask,
    )

    return voted_binary, combined_score, candidates, thresholds


# ============================================================
# 时间连续性跟踪
# ============================================================

def bbox_iou(
    first: Tuple[int, int, int, int],
    second: Tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0,
        intersection_x2
        - intersection_x1,
    )

    intersection_height = max(
        0,
        intersection_y2
        - intersection_y1,
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    first_area = max(
        1,
        (ax2 - ax1)
        * (ay2 - ay1),
    )

    second_area = max(
        1,
        (bx2 - bx1)
        * (by2 - by1),
    )

    return intersection / max(
        first_area
        + second_area
        - intersection,
        1,
    )


def bbox_center_distance(
    first: Tuple[int, int, int, int],
    second: Tuple[int, int, int, int],
) -> Tuple[float, float]:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second

    first_center = np.asarray(
        [
            (ax1 + ax2) / 2.0,
            (ay1 + ay2) / 2.0,
        ],
        dtype=np.float32,
    )

    second_center = np.asarray(
        [
            (bx1 + bx2) / 2.0,
            (by1 + by2) / 2.0,
        ],
        dtype=np.float32,
    )

    distance = float(
        np.linalg.norm(
            first_center
            - second_center
        )
    )

    scale = max(
        math.hypot(
            ax2 - ax1,
            ay2 - ay1,
        ),
        math.hypot(
            bx2 - bx1,
            by2 - by1,
        ),
        1.0,
    )

    return distance, scale


class TemporalAnomalyTracker:
    def __init__(self):
        self.tracks: List[
            AnomalyTrack
        ] = []
        self.next_track_id = 1

    def update(
        self,
        candidates: List[
            CandidateRegion
        ],
        timestamp: float,
    ) -> None:
        for track in self.tracks:
            track.matched_this_update = False

        candidate_pairs = []

        for track_index, track in enumerate(
            self.tracks
        ):
            for candidate_index, candidate in enumerate(
                candidates
            ):
                iou = bbox_iou(
                    track.bbox,
                    candidate.bbox,
                )

                distance, scale = (
                    bbox_center_distance(
                        track.bbox,
                        candidate.bbox,
                    )
                )

                close_enough = (
                    distance
                    <= scale
                    * TRACK_CENTER_DISTANCE_RATIO
                )

                if (
                    iou >= TRACK_IOU_THRESHOLD
                    or close_enough
                ):
                    cost = (
                        (1.0 - iou) * 100.0
                        + distance
                    )

                    candidate_pairs.append(
                        (
                            cost,
                            track_index,
                            candidate_index,
                        )
                    )

        candidate_pairs.sort(
            key=lambda item: item[0]
        )

        used_tracks = set()
        used_candidates = set()

        for (
            _,
            track_index,
            candidate_index,
        ) in candidate_pairs:
            if (
                track_index in used_tracks
                or candidate_index
                in used_candidates
            ):
                continue

            track = self.tracks[
                track_index
            ]

            candidate = candidates[
                candidate_index
            ]

            old_box = np.asarray(
                track.bbox,
                dtype=np.float32,
            )

            new_box = np.asarray(
                candidate.bbox,
                dtype=np.float32,
            )

            smoothed = (
                0.58 * old_box
                + 0.42 * new_box
            )

            track.bbox = tuple(
                int(round(value))
                for value in smoothed
            )

            track.last_seen = timestamp
            track.hits += 1
            track.misses = 0
            track.mean_score = (
                0.70 * track.mean_score
                + 0.30
                * candidate.mean_score
                if track.hits > 1
                else candidate.mean_score
            )

            track.vote_mean = (
                0.70 * track.vote_mean
                + 0.30
                * candidate.vote_mean
                if track.hits > 1
                else candidate.vote_mean
            )

            track.matched_this_update = True

            if (
                track.hits
                >= CONFIRM_MIN_HITS
                and track.duration
                >= CONFIRM_MIN_SECONDS
            ):
                track.confirmed = True

            used_tracks.add(
                track_index
            )

            used_candidates.add(
                candidate_index
            )

        for index, track in enumerate(
            self.tracks
        ):
            if index not in used_tracks:
                track.misses += 1
                track.matched_this_update = False

        for index, candidate in enumerate(
            candidates
        ):
            if index in used_candidates:
                continue

            self.tracks.append(
                AnomalyTrack(
                    track_id=self.next_track_id,
                    bbox=candidate.bbox,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    hits=1,
                    misses=0,
                    confirmed=False,
                    mean_score=candidate.mean_score,
                    vote_mean=candidate.vote_mean,
                )
            )

            self.next_track_id += 1

        self.tracks = [
            track
            for track in self.tracks
            if track.misses
            <= MAX_TRACK_MISSES
        ]

    def active_tracks(
        self,
    ) -> List[AnomalyTrack]:
        return [
            track
            for track in self.tracks
            if track.misses
            <= MAX_TRACK_MISSES
        ]


# ============================================================
# 显示
# ============================================================

def create_debug_inset(
    voted_mask: np.ndarray,
    score: np.ndarray,
    vehicle_mask: np.ndarray,
    width: int,
) -> np.ndarray:
    height, source_width = (
        voted_mask.shape[:2]
    )

    panel = np.zeros(
        (height, source_width, 3),
        dtype=np.uint8,
    )

    normalized_score = np.clip(
        score,
        0,
        100,
    )

    normalized_score = (
        normalized_score / 100.0
        * 255.0
    ).astype(np.uint8)

    heat = cv2.applyColorMap(
        normalized_score,
        cv2.COLORMAP_JET,
    )

    panel = heat
    panel[voted_mask > 0] = (
        0,
        0,
        255,
    )
    panel[vehicle_mask > 0] = (
        255,
        80,
        20,
    )

    scale = min(
        1.0,
        width / max(
            source_width,
            1,
        ),
    )

    panel = cv2.resize(
        panel,
        (
            max(
                1,
                int(
                    round(
                        source_width
                        * scale
                    )
                ),
            ),
            max(
                1,
                int(
                    round(
                        height
                        * scale
                    )
                ),
            ),
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    cv2.rectangle(
        panel,
        (0, 0),
        (
            panel.shape[1] - 1,
            panel.shape[0] - 1,
        ),
        (245, 245, 245),
        2,
    )

    cv2.putText(
        panel,
        "Difference / vote mask",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return panel


def draw_result(
    frame: np.ndarray,
    road_polygon: np.ndarray,
    road_mask: np.ndarray,
    vehicle_mask: np.ndarray,
    accepted_detections: Sequence[Any],
    tracks: List[AnomalyTrack],
    voted_mask: np.ndarray,
    combined_score: np.ndarray,
    selected_references: List[
        AlignedReference
    ],
    thresholds: List[float],
    inference_ms: float,
    processing_fps: float,
    frame_index: int,
    total_frames: int,
    paused: bool,
) -> np.ndarray:
    output = frame.copy()

    # 道路区域。
    road_layer = output.copy()
    road_layer[road_mask > 0] = (
        45,
        150,
        90,
    )

    output = cv2.addWeighted(
        road_layer,
        0.10,
        output,
        0.90,
        0,
    )

    polygon_int = np.round(
        road_polygon
    ).astype(np.int32).reshape(
        (-1, 1, 2)
    )

    cv2.polylines(
        output,
        [polygon_int],
        True,
        (0, 225, 255),
        4,
        cv2.LINE_AA,
    )

    # 当前车辆忽略区域。
    vehicle_layer = output.copy()
    vehicle_layer[vehicle_mask > 0] = (
        35,
        35,
        230,
    )

    output = cv2.addWeighted(
        vehicle_layer,
        0.25,
        output,
        0.75,
        0,
    )

    for accepted in accepted_detections:
        detection = accepted[0]
        x1, y1, x2, y2 = (
            detection.bbox
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (40, 220, 255),
            2,
            cv2.LINE_AA,
        )

    # 时间异常轨迹。
    confirmed_count = 0

    for track in tracks:
        x1, y1, x2, y2 = track.bbox

        if track.confirmed:
            confirmed_count += 1
            color = (20, 20, 245)
            label_prefix = "ANOMALY"
            thickness = 4
        else:
            color = (0, 165, 255)
            label_prefix = "candidate"
            thickness = 2

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
            cv2.LINE_AA,
        )

        label = (
            f"{label_prefix} #{track.track_id} "
            f"{track.duration:.1f}s "
            f"score={track.mean_score:.0f}"
        )

        label_y = max(
            24,
            y1 - 8,
        )

        cv2.putText(
            output,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            1,
            cv2.LINE_AA,
        )

    inset = create_debug_inset(
        voted_mask,
        combined_score,
        vehicle_mask,
        width=max(
            270,
            int(
                round(
                    output.shape[1]
                    * 0.28
                )
            ),
        ),
    )

    inset_x = (
        output.shape[1]
        - inset.shape[1]
        - 15
    )
    inset_y = 115

    if (
        inset_x >= 0
        and inset_y + inset.shape[0]
        <= output.shape[0]
    ):
        output[
            inset_y:inset_y
            + inset.shape[0],
            inset_x:inset_x
            + inset.shape[1],
        ] = inset

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 104),
        (9, 16, 25),
        -1,
    )

    state = (
        "PAUSED"
        if paused
        else "PLAYING"
    )

    reference_names = ",".join(
        item.entry.name.replace(
            "reference_",
            "R",
        ).replace(".png", "")
        for item in selected_references
    ) or "none"

    threshold_text = (
        ",".join(
            f"{value:.0f}"
            for value in thresholds
        )
        if thresholds
        else "-"
    )

    candidate_count = sum(
        1
        for track in tracks
        if not track.confirmed
    )

    line1 = (
        f"{state} | Frame "
        f"{frame_index}/"
        f"{total_frames if total_frames > 0 else '?'} "
        f"| FPS {processing_fps:.1f}"
    )

    line2 = (
        f"normal.onnx infer "
        f"{inference_ms:.1f} ms "
        f"| references {reference_names} "
        f"| thresholds {threshold_text}"
    )

    line3 = (
        f"vehicles {len(accepted_detections)} "
        f"| candidates {candidate_count} "
        f"| confirmed anomalies "
        f"{confirmed_count}"
    )

    for text, y, color in (
        (
            line1,
            28,
            (245, 245, 245),
        ),
        (
            line2,
            61,
            (80, 230, 255),
        ),
        (
            line3,
            93,
            (120, 240, 150),
        ),
    ):
        cv2.putText(
            output,
            text,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    project_root = Path(
        __file__
    ).resolve().parent

    input_video_name = (
        sys.argv[1]
        if len(sys.argv) >= 2
        else DEFAULT_VIDEO_FILENAME
    )

    input_video_path = (
        Path(input_video_name)
        if Path(input_video_name).is_absolute()
        else project_root
        / input_video_name
    )

    model_path = (
        project_root / MODEL_FILENAME
    )

    anchor_image_path = (
        project_root
        / ANCHOR_IMAGE_PATH
    )

    anchor_roi_json_path = (
        project_root
        / ANCHOR_ROI_JSON_PATH
    )

    anchor_road_mask_path = (
        project_root
        / ANCHOR_ROAD_MASK_PATH
    )

    output_video_path = (
        project_root
        / OUTPUT_VIDEO_PATH
    )

    for path, label in (
        (
            input_video_path,
            "输入视频",
        ),
        (
            model_path,
            "normal.onnx",
        ),
        (
            anchor_image_path,
            "reference_01.png",
        ),
        (
            anchor_roi_json_path,
            "road_roi.json",
        ),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"没有找到{label}：{path}"
            )

    anchor_image = imread_unicode(
        anchor_image_path,
        cv2.IMREAD_COLOR,
    )

    if anchor_image is None:
        raise RuntimeError(
            f"无法读取主基准图："
            f"{anchor_image_path}"
        )

    anchor_height, anchor_width = (
        anchor_image.shape[:2]
    )

    anchor_polygon = load_roi_points(
        anchor_roi_json_path
    )

    anchor_road_mask = None

    if anchor_road_mask_path.exists():
        anchor_road_mask = imread_unicode(
            anchor_road_mask_path,
            cv2.IMREAD_GRAYSCALE,
        )

    if anchor_road_mask is None:
        anchor_road_mask = polygon_to_mask(
            anchor_polygon,
            anchor_width,
            anchor_height,
        )

    anchor_road_mask = cv2.resize(
        anchor_road_mask,
        (
            anchor_width,
            anchor_height,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    detector = (
        NormalOnnxVehicleDetector(
            model_path
        )
    )

    actual_providers = list(detector.session.get_providers())
    print(f"[CUDA] normal.onnx 实际 Provider：{actual_providers}")
    if not actual_providers or actual_providers[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            "normal.onnx 没有成功使用GPU，"
            f"实际 Provider：{actual_providers}"
        )

    road_roi_tracker = RoadRoiTracker(
        anchor_image=anchor_image,
        anchor_polygon=anchor_polygon,
        anchor_road_mask=anchor_road_mask,
    )

    reference_bank = ReferenceBank(
        project_root=project_root,
        detector=detector,
        anchor_size=(
            anchor_width,
            anchor_height,
        ),
    )

    capture, temp_video_path = (
        open_video_with_fallback(
            input_video_path
        )
    )

    writer = None

    try:
        source_fps = float(
            capture.get(
                cv2.CAP_PROP_FPS
            )
        )

        if (
            not np.isfinite(source_fps)
            or source_fps <= 1.0
        ):
            source_fps = 25.0

        total_frames = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if SAVE_OUTPUT_VIDEO:
            output_video_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            writer = cv2.VideoWriter(
                str(output_video_path),
                cv2.VideoWriter_fourcc(
                    *"mp4v"
                ),
                source_fps,
                (
                    anchor_width,
                    anchor_height,
                ),
            )

            if not writer.isOpened():
                raise RuntimeError(
                    "无法创建输出视频："
                    f"{output_video_path}"
                )

        print("=" * 80)
        print("多基准道路障碍物检测 · GPU V4 直接精确配准")
        print(
            f"输入视频："
            f"{input_video_path}"
        )
        print(
            f"可用基准："
            f"{len(reference_bank.entries)}"
        )
        print(
            "核心逻辑：每张候选基准直接ORB/Homography；"
            "禁止离线矩阵组合推算"
        )
        print(
            f"并行线程：配准={ALIGNMENT_WORKERS}，"
            f"精确差分={DIFFERENCE_WORKERS}"
        )
        print(
            f"输出视频："
            f"{output_video_path}"
        )
        print("=" * 80)

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        temporal_tracker = (
            TemporalAnomalyTracker()
        )

        paused = False
        single_step = False
        frame_index = 0

        last_frame: Optional[
            np.ndarray
        ] = None

        last_road_polygon = (
            anchor_polygon.copy()
        )

        last_current_detections = []
        last_inference_ms = 0.0
        last_vehicle_mask = np.zeros(
            (
                anchor_height,
                anchor_width,
            ),
            dtype=np.uint8,
        )
        last_accepted_detections = []

        last_aligned_references: List[
            AlignedReference
        ] = []

        last_voted_mask = np.zeros(
            (
                anchor_height,
                anchor_width,
            ),
            dtype=np.uint8,
        )

        last_combined_score = np.zeros(
            (
                anchor_height,
                anchor_width,
            ),
            dtype=np.float32,
        )

        last_thresholds: List[
            float
        ] = []

        processing_fps_ema = 0.0

        while True:
            should_read = (
                not paused
                or single_step
                or last_frame is None
            )

            if should_read:
                success, source_frame = (
                    capture.read()
                )

                if (
                    not success
                    or source_frame is None
                ):
                    print("视频播放结束。")
                    break

                frame_index += 1

                frame = cv2.resize(
                    source_frame,
                    (
                        anchor_width,
                        anchor_height,
                    ),
                    interpolation=cv2.INTER_AREA,
                )

                last_frame = frame
                started = time.perf_counter()

                (
                    last_road_polygon,
                    _,
                    _,
                ) = road_roi_tracker.update(
                    frame
                )

                current_road_mask = (
                    polygon_to_mask(
                        last_road_polygon,
                        anchor_width,
                        anchor_height,
                    )
                )

                if (
                    frame_index == 1
                    or frame_index
                    % YOLO_EVERY_N_FRAMES
                    == 0
                ):
                    (
                        last_current_detections,
                        last_inference_ms,
                    ) = detector.detect(frame)

                (
                    last_vehicle_mask,
                    last_accepted_detections,
                ) = build_vehicle_mask(
                    last_current_detections,
                    detector,
                    current_road_mask,
                )

                if (
                    frame_index == 1
                    or frame_index
                    % ANOMALY_EVERY_N_FRAMES
                    == 0
                ):
                    last_aligned_references = (
                        reference_bank.align_best(
                            frame,
                            last_vehicle_mask,
                        )
                    )

                    (
                        last_voted_mask,
                        last_combined_score,
                        candidates,
                        last_thresholds,
                    ) = (
                        run_multi_reference_difference(
                            current_frame=frame,
                            current_road_mask=current_road_mask,
                            current_vehicle_mask=last_vehicle_mask,
                            aligned_references=last_aligned_references,
                        )
                    )

                    current_timestamp = (
                        frame_index
                        / source_fps
                    )

                    temporal_tracker.update(
                        candidates,
                        current_timestamp,
                    )

                elapsed = (
                    time.perf_counter()
                    - started
                )

                instant_fps = (
                    1.0 / elapsed
                    if elapsed > 0
                    else 0.0
                )

                if processing_fps_ema <= 0:
                    processing_fps_ema = (
                        instant_fps
                    )
                else:
                    processing_fps_ema = (
                        0.90
                        * processing_fps_ema
                        + 0.10
                        * instant_fps
                    )

                result = draw_result(
                    frame=frame,
                    road_polygon=last_road_polygon,
                    road_mask=current_road_mask,
                    vehicle_mask=last_vehicle_mask,
                    accepted_detections=last_accepted_detections,
                    tracks=temporal_tracker.active_tracks(),
                    voted_mask=last_voted_mask,
                    combined_score=last_combined_score,
                    selected_references=last_aligned_references,
                    thresholds=last_thresholds,
                    inference_ms=last_inference_ms,
                    processing_fps=processing_fps_ema,
                    frame_index=frame_index,
                    total_frames=total_frames,
                    paused=paused,
                )

                if writer is not None:
                    writer.write(result)

                single_step = False

            else:
                current_road_mask = (
                    polygon_to_mask(
                        last_road_polygon,
                        anchor_width,
                        anchor_height,
                    )
                )

                result = draw_result(
                    frame=last_frame,
                    road_polygon=last_road_polygon,
                    road_mask=current_road_mask,
                    vehicle_mask=last_vehicle_mask,
                    accepted_detections=last_accepted_detections,
                    tracks=temporal_tracker.active_tracks(),
                    voted_mask=last_voted_mask,
                    combined_score=last_combined_score,
                    selected_references=last_aligned_references,
                    thresholds=last_thresholds,
                    inference_ms=last_inference_ms,
                    processing_fps=processing_fps_ema,
                    frame_index=frame_index,
                    total_frames=total_frames,
                    paused=True,
                )

            display = resize_to_fit(
                result,
                MAX_DISPLAY_WIDTH,
                MAX_DISPLAY_HEIGHT,
            )

            cv2.imshow(
                WINDOW_NAME,
                display,
            )

            delay = (
                max(
                    1,
                    int(
                        round(
                            1000.0
                            / source_fps
                        )
                    ),
                )
                if not paused
                else 30
            )

            key = cv2.waitKey(
                delay
            )

            key_code = (
                -1
                if key == -1
                else key & 0xFF
            )

            if key_code in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break

            if key_code == ord(" "):
                paused = not paused

            elif key_code in (
                ord("s"),
                ord("S"),
            ):
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

    finally:
        capture.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

        if temp_video_path is not None:
            temp_video_path.unlink(
                missing_ok=True
            )

    print("=" * 80)
    print("处理结束")
    print(
        f"结果视频："
        f"{output_video_path}"
    )
    print("=" * 80)
    print(
        "橙色框表示短时异常候选；"
        "红色框表示连续存在并确认的道路障碍物。"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        sys.exit(0)
    except Exception as exc:
        cv2.destroyAllWindows()
        print(
            f"[失败] "
            f"{type(exc).__name__}: "
            f"{exc}"
        )
        raise

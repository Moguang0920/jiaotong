# -*- coding: utf-8 -*-
"""
多基准道路障碍物检测 · GPU 像素流水线版

相对旧版的主要优化：
1. normal.onnx 继续使用 ONNX Runtime CUDAExecutionProvider。
2. 不再每次对 7 张基准图重复执行 ORB/Homography：
   - 只用 RoadRoiTracker 对主基准图做一次相机位姿跟随；
   - 使用 road_mask_mapping.json 中“主基准 -> 各基准”的离线矩阵；
   - 组合得到“各基准 -> 当前帧”的 Homography。
3. 三张基准图的透视变换、车辆/道路 Mask、差分、投票和形态学，
   全部组成一个 PyTorch CUDA batch，在 GPU 上一次完成。
4. ORB 道路跟随降频执行，中间帧复用最近一次 Homography。
5. 视频保存优先调用 FFmpeg h264_nvenc；不可用时回退 OpenCV mp4v。
6. 只有最终二值 Mask 和差异得分返回 CPU，连通域与少量框跟踪保留 CPU。

运行：
    python multi_reference_road_anomaly_demo.py 异常物品.mp4

依赖：
    normal.onnx
    road_roi_vehicle_mask_demo.py
    road_anomaly_data/camera_01/reference_bank/
    road_anomaly_data/camera_01/road_masks/
    road_anomaly_data/camera_01/road_roi/
    PyTorch CUDA
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


GPU_PIPELINE_VERSION = "2026-07-14-GPU-V3-DLL-ORDER-FIX"
print("=" * 80)
print(f"[启动版本] {GPU_PIPELINE_VERSION}")
print(f"[启动文件] {Path(__file__).resolve()}")
print("=" * 80)

# ============================================================
# CUDA DLL 初始化
#
# Windows 下必须先导入 PyTorch：
# PyTorch 会优先加载与自身版本匹配的 CUDA/cuDNN DLL。
# 然后 ONNX Runtime 使用已经加载的同一套 DLL。
#
# 不能先执行 ort.preload_dlls(directory="")，否则 NVIDIA
# site-packages 中另一套 cuDNN 可能先进入进程，导致 Torch
# 加载 cudnn_cnn64_9.dll 时出现 WinError 127。
# ============================================================

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:
    raise RuntimeError(
        "PyTorch CUDA 导入失败。"
        "请在当前项目 .venv 中运行 GPU 安装脚本。"
        f"\n原始错误：{exc}"
    ) from exc

if not torch.cuda.is_available():
    raise RuntimeError(
        "PyTorch 已导入，但 torch.cuda.is_available() 为 False。"
        f"\nPyTorch：{torch.__version__}"
        f"\nPyTorch CUDA：{getattr(torch.version, 'cuda', None)}"
    )

# 强制初始化 CUDA Context，确保 Torch DLL 已经完整加载。
try:
    _cuda_test_tensor = torch.zeros(
        (1,),
        device="cuda",
        dtype=torch.float32,
    )
    torch.cuda.synchronize()
    del _cuda_test_tensor
except Exception as exc:
    raise RuntimeError(
        "PyTorch CUDA Context 初始化失败："
        f"{exc}"
    ) from exc

torch.backends.cudnn.benchmark = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

print(
    "[CUDA] PyTorch 已优先加载 CUDA/cuDNN："
    f"torch={torch.__version__} | "
    f"CUDA={torch.version.cuda} | "
    f"GPU={torch.cuda.get_device_name(0)}"
)

try:
    import onnxruntime as ort
except Exception as exc:
    raise RuntimeError(
        f"onnxruntime-gpu 导入失败：{exc}"
    ) from exc

# 使用默认搜索顺序 directory=None：
# Windows 下优先搜索 PyTorch 的 torch/lib，再搜索 NVIDIA site-packages。
if hasattr(ort, "preload_dlls"):
    try:
        ort.preload_dlls(
            cuda=True,
            cudnn=True,
            msvc=True,
            directory=None,
        )
    except Exception as exc:
        raise RuntimeError(
            "ONNX Runtime 复用 PyTorch CUDA/cuDNN DLL 失败："
            f"{exc}"
        ) from exc

_AVAILABLE_PROVIDERS = list(
    ort.get_available_providers()
)

print(
    "[CUDA] ONNX Runtime 已复用 PyTorch DLL；Provider："
    f"{_AVAILABLE_PROVIDERS}"
)

if "CUDAExecutionProvider" not in _AVAILABLE_PROVIDERS:
    raise RuntimeError(
        "当前环境没有 CUDAExecutionProvider："
        f"{_AVAILABLE_PROVIDERS}"
    )


# ============================================================
# 复用现有车辆检测与道路跟随
# ============================================================

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
        "无法导入 road_roi_vehicle_mask_demo.py。"
        "请确认它位于项目根目录。"
        f"\n原始错误：{exc}"
    ) from exc


# ============================================================
# 路径
# ============================================================

DEFAULT_VIDEO_FILENAME = "正常道路.mp4"
MODEL_FILENAME = "normal.onnx"

CAMERA_ROOT = Path("road_anomaly_data") / "camera_01"
REFERENCE_DIR = CAMERA_ROOT / "reference_bank"
ROAD_MASK_DIR = CAMERA_ROOT / "road_masks"
ROAD_MAPPING_JSON = CAMERA_ROOT / "road_mask_mapping.json"

ANCHOR_IMAGE_PATH = REFERENCE_DIR / "reference_01.png"
ANCHOR_ROI_JSON_PATH = CAMERA_ROOT / "road_roi" / "road_roi.json"
ANCHOR_ROAD_MASK_PATH = CAMERA_ROOT / "road_roi" / "road_mask.png"

REFERENCE_VEHICLE_MASK_DIR = CAMERA_ROOT / "reference_vehicle_masks"
REFERENCE_VEHICLE_PREVIEW_DIR = CAMERA_ROOT / "reference_vehicle_previews"
REFERENCE_CACHE_JSON = CAMERA_ROOT / "reference_runtime_cache_gpu.json"

OUTPUT_VIDEO_PATH = CAMERA_ROOT / "road_anomaly_result.mp4"


# ============================================================
# 参数
# ============================================================

WINDOW_NAME = "Multi-reference Road Anomaly Detection - GPU Pipeline"
SAVE_OUTPUT_VIDEO = True

MAX_DISPLAY_WIDTH = 1750
MAX_DISPLAY_HEIGHT = 950

# 车辆检测：GPU ONNX
YOLO_EVERY_N_FRAMES = 1

# 主基准 ORB/Homography：CPU，但只隔几帧更新一次。
ROAD_ROI_UPDATE_EVERY_N_FRAMES = 3

# GPU 多基准差分频率。
ANOMALY_EVERY_N_FRAMES = 2

# 每次选择最接近的三张基准图。
TOP_REFERENCE_COUNT = 3
MIN_REFERENCE_VOTES = 2

# 基准图中的车辆 Mask 缓存。
REBUILD_REFERENCE_VEHICLE_MASKS = False
VEHICLE_EXTRA_DILATE = 7
ROAD_BOUNDARY_ERODE = 5

# GPU 差异阈值。
BASE_DIFF_THRESHOLD = 37.0
MAX_ADAPTIVE_THRESHOLD = 72.0
MAD_MULTIPLIER = 4.2
MAD_EXTRA = 7.0
THRESHOLD_SAMPLE_STRIDE = 4

# GPU 形态学。
OPEN_KERNEL_SIZE = 3
CLOSE_KERNEL_SIZE = 9

# 候选区域。
BOTTOM_MIN_AREA = 170
TOP_MIN_AREA = 34
MAX_COMPONENT_ROAD_RATIO = 0.16
MIN_COMPONENT_MEAN_SCORE = 42.0

# 时间确认。
CONFIRM_MIN_SECONDS = 1.25
CONFIRM_MIN_HITS = 5
MAX_TRACK_MISSES = 8
TRACK_IOU_THRESHOLD = 0.10
TRACK_CENTER_DISTANCE_RATIO = 1.25

# 离线矩阵合理性。
MIN_WARP_AREA_RATIO = 0.65
MAX_WARP_AREA_RATIO = 1.45
MAX_CENTER_SHIFT_RATIO = 0.32
MAX_PERSPECTIVE_TERM = 0.008

# GPU 设置。
GPU_DEVICE_INDEX = 0
GPU_USE_TF32 = True

# 视频编码。
PREFER_NVENC = True
REQUIRE_NVENC = False
NVENC_PRESET = "p4"
NVENC_CQ = 23


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
    anchor_to_reference: np.ndarray
    gpu_image: Optional[torch.Tensor] = None
    gpu_road_mask: Optional[torch.Tensor] = None
    gpu_vehicle_mask: Optional[torch.Tensor] = None


@dataclass
class AlignedReference:
    entry: ReferenceEntry
    homography: np.ndarray
    source: str = "offline_mapping"


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
# 文件工具
# ============================================================

def imwrite_unicode(
    path: Path,
    image: np.ndarray,
    params: Optional[List[int]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image, params or [])
    if not ok:
        raise RuntimeError(f"图片编码失败：{path}")
    encoded.tofile(str(path))


def as_homography(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape == (2, 3):
        value = np.vstack(
            [value, np.asarray([0.0, 0.0, 1.0], dtype=np.float64)]
        )
    if value.shape != (3, 3):
        raise ValueError(f"矩阵尺寸不是 2x3 或 3x3：{value.shape}")
    if abs(float(value[2, 2])) > 1e-12:
        value = value / float(value[2, 2])
    return value


# ============================================================
# 快速场景特征
# ============================================================

def build_scene_feature(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    small = cv2.resize(
        gray,
        (32, 18),
        interpolation=cv2.INTER_AREA,
    )
    equalized = cv2.equalizeHist(small)
    edges = cv2.Canny(equalized, 45, 130)

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [16],
        [0, 256],
    ).reshape(-1)

    histogram = histogram / max(float(histogram.sum()), 1.0)

    return np.concatenate(
        [
            equalized.reshape(-1).astype(np.float32) / 255.0,
            edges.reshape(-1).astype(np.float32) / 255.0 * 0.75,
            histogram.astype(np.float32) * 4.0,
        ]
    ).astype(np.float32)


# ============================================================
# 基准库：离线矩阵 + GPU 缓存
# ============================================================

class ReferenceBank:
    def __init__(
        self,
        project_root: Path,
        detector: NormalOnnxVehicleDetector,
        anchor_image: np.ndarray,
        anchor_size: Tuple[int, int],
    ):
        self.project_root = project_root
        self.detector = detector
        self.anchor_image = anchor_image
        self.width, self.height = anchor_size
        self.entries: List[ReferenceEntry] = []

        self._mapping_records = self._load_mapping_records()
        self._orb = cv2.ORB_create(
            nfeatures=5000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=21,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=12,
        )
        self._matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=False,
        )
        self._anchor_gray = self._normalize_gray(anchor_image)
        (
            self._anchor_keypoints,
            self._anchor_descriptors,
        ) = self._orb.detectAndCompute(self._anchor_gray, None)

        self._load_all()

    @staticmethod
    def _normalize_gray(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        return clahe.apply(gray)

    def _load_mapping_records(self) -> Dict[str, np.ndarray]:
        path = self.project_root / ROAD_MAPPING_JSON
        records: Dict[str, np.ndarray] = {
            "reference_01.png": np.eye(3, dtype=np.float64)
        }

        if not path.exists():
            print(
                "[映射] road_mask_mapping.json 不存在，"
                "缺失矩阵将在启动阶段自动估计。"
            )
            return records

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[映射] 配置读取失败，将自动估计：{exc}")
            return records

        for item in data.get("records", []):
            if item.get("status") != "success":
                continue
            name = str(item.get("reference", "")).strip()
            matrix = item.get("matrix")
            if not name or matrix is None:
                continue
            try:
                records[name] = as_homography(matrix)
            except Exception:
                continue

        print(f"[映射] 已载入离线矩阵：{len(records)} 张")
        return records

    def _load_road_mask(
        self,
        reference_path: Path,
    ) -> Optional[np.ndarray]:
        path = (
            self.project_root
            / ROAD_MASK_DIR
            / f"{reference_path.stem}_road_mask.png"
        )

        if path.exists():
            value = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
            if value is not None:
                return value

        if reference_path.name == "reference_01.png":
            path = self.project_root / ANCHOR_ROAD_MASK_PATH
            if path.exists():
                return imread_unicode(path, cv2.IMREAD_GRAYSCALE)

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
        mask_path, preview_path = self._reference_vehicle_paths(
            reference_path
        )

        if mask_path.exists() and not REBUILD_REFERENCE_VEHICLE_MASKS:
            cached = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
            if cached is not None and cached.shape[:2] == road_mask.shape[:2]:
                _, cached = cv2.threshold(
                    cached,
                    127,
                    255,
                    cv2.THRESH_BINARY,
                )
                return cached

        detections, inference_ms = self.detector.detect(image)
        vehicle_mask, accepted = build_vehicle_mask(
            detections,
            self.detector,
            road_mask,
        )

        if VEHICLE_EXTRA_DILATE > 0:
            kernel = np.ones(
                (VEHICLE_EXTRA_DILATE, VEHICLE_EXTRA_DILATE),
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

        imwrite_unicode(mask_path, vehicle_mask)

        preview = image.copy()
        overlay = preview.copy()
        overlay[road_mask > 0] = (50, 170, 95)
        preview = cv2.addWeighted(overlay, 0.13, preview, 0.87, 0)

        overlay = preview.copy()
        overlay[vehicle_mask > 0] = (30, 30, 235)
        preview = cv2.addWeighted(overlay, 0.35, preview, 0.65, 0)

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
                f"{reference_path.name} | vehicles={len(accepted)} "
                f"| infer={inference_ms:.1f}ms"
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

    def _estimate_anchor_to_reference(
        self,
        target: np.ndarray,
    ) -> Optional[np.ndarray]:
        if self._anchor_descriptors is None:
            return None

        target_gray = self._normalize_gray(target)
        target_keypoints, target_descriptors = self._orb.detectAndCompute(
            target_gray,
            None,
        )

        if target_descriptors is None or len(target_keypoints) < 20:
            return None

        pairs = self._matcher.knnMatch(
            self._anchor_descriptors,
            target_descriptors,
            k=2,
        )

        good = []
        for pair in pairs:
            if len(pair) != 2:
                continue
            first, second = pair
            if first.distance < 0.76 * second.distance:
                good.append(first)

        if len(good) < 22:
            return None

        src = np.float32(
            [self._anchor_keypoints[item.queryIdx].pt for item in good]
        ).reshape(-1, 1, 2)
        dst = np.float32(
            [target_keypoints[item.trainIdx].pt for item in good]
        ).reshape(-1, 1, 2)

        matrix, inliers = cv2.findHomography(
            src,
            dst,
            cv2.RANSAC,
            3.5,
            maxIters=4000,
            confidence=0.995,
        )

        if matrix is None or inliers is None:
            return None

        inlier_count = int(inliers.reshape(-1).sum())
        if inlier_count < 14:
            return None

        return as_homography(matrix)

    def _load_all(self) -> None:
        reference_dir = self.project_root / REFERENCE_DIR
        paths = sorted(reference_dir.glob("reference_*.png"))

        if not paths:
            raise FileNotFoundError(
                f"基准目录中没有 reference_*.png：{reference_dir}"
            )

        cache_records = []

        print("=" * 80)
        print("正在准备 GPU 多基准库")
        print(f"基准图数量：{len(paths)}")
        print("=" * 80)

        for index, path in enumerate(paths, start=1):
            image = imread_unicode(path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"[跳过] 无法读取：{path.name}")
                continue

            image = cv2.resize(
                image,
                (self.width, self.height),
                interpolation=cv2.INTER_AREA,
            )

            road_mask = self._load_road_mask(path)
            if road_mask is None:
                print(f"[跳过] 缺少道路 Mask：{path.name}")
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

            vehicle_mask = self._create_reference_vehicle_mask(
                image,
                road_mask,
                path,
            )

            anchor_to_reference = self._mapping_records.get(path.name)

            if anchor_to_reference is None:
                print(f"[映射] 启动时估计：{path.name}")
                anchor_to_reference = self._estimate_anchor_to_reference(image)

            if anchor_to_reference is None:
                print(f"[跳过] 无法得到离线映射：{path.name}")
                continue

            entry = ReferenceEntry(
                name=path.name,
                image_path=path,
                image=image,
                road_mask=road_mask,
                vehicle_mask=vehicle_mask,
                scene_feature=build_scene_feature(image),
                anchor_to_reference=anchor_to_reference,
            )
            self.entries.append(entry)

            cache_records.append(
                {
                    "reference": path.name,
                    "road_pixels": int(np.count_nonzero(road_mask)),
                    "vehicle_pixels": int(np.count_nonzero(vehicle_mask)),
                    "anchor_to_reference": anchor_to_reference.tolist(),
                }
            )

            print(
                f"[{index}/{len(paths)}] {path.name}："
                "映射与车辆 Mask 已就绪"
            )

        if len(self.entries) < TOP_REFERENCE_COUNT:
            raise RuntimeError(
                f"可用基准图不足：{len(self.entries)}"
            )

        cache_path = self.project_root / REFERENCE_CACHE_JSON
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "reference_count": len(self.entries),
                    "records": cache_records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print("=" * 80)
        print(f"GPU 基准库准备完成：{len(self.entries)} 张")
        print("=" * 80)

    def _validate_homography(self, matrix: np.ndarray) -> bool:
        corners = np.asarray(
            [
                [0.0, 0.0],
                [self.width - 1.0, 0.0],
                [self.width - 1.0, self.height - 1.0],
                [0.0, self.height - 1.0],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        try:
            transformed = cv2.perspectiveTransform(
                corners,
                matrix.astype(np.float64),
            ).reshape(-1, 2)
        except Exception:
            return False

        if not np.isfinite(transformed).all():
            return False

        contour = np.round(transformed).astype(np.int32).reshape(-1, 1, 2)
        if not cv2.isContourConvex(contour):
            return False

        original_area = float(self.width * self.height)
        transformed_area = abs(
            float(cv2.contourArea(transformed.astype(np.float32)))
        )
        area_ratio = transformed_area / max(original_area, 1.0)

        source_center = np.asarray(
            [self.width / 2.0, self.height / 2.0],
            dtype=np.float32,
        )
        target_center = transformed.mean(axis=0)
        shift = float(np.linalg.norm(target_center - source_center))
        max_shift = (
            math.hypot(self.width, self.height)
            * MAX_CENTER_SHIFT_RATIO
        )

        perspective = max(
            abs(float(matrix[2, 0])),
            abs(float(matrix[2, 1])),
        )

        return bool(
            MIN_WARP_AREA_RATIO <= area_ratio <= MAX_WARP_AREA_RATIO
            and shift <= max_shift
            and perspective <= MAX_PERSPECTIVE_TERM
        )

    def select(
        self,
        current_frame: np.ndarray,
        anchor_to_current: np.ndarray,
    ) -> List[AlignedReference]:
        feature = build_scene_feature(current_frame)

        scored: List[Tuple[float, ReferenceEntry]] = []
        for entry in self.entries:
            distance = float(
                np.mean(np.abs(feature - entry.scene_feature))
            )
            scored.append((distance, entry))

        scored.sort(key=lambda item: item[0])

        selected: List[AlignedReference] = []

        for _, entry in scored:
            try:
                reference_to_current = (
                    as_homography(anchor_to_current)
                    @ np.linalg.inv(entry.anchor_to_reference)
                )
                reference_to_current = as_homography(reference_to_current)
            except Exception:
                continue

            if not self._validate_homography(reference_to_current):
                continue

            selected.append(
                AlignedReference(
                    entry=entry,
                    homography=reference_to_current,
                )
            )

            if len(selected) >= TOP_REFERENCE_COUNT:
                break

        return selected


# ============================================================
# GPU 像素流水线
# ============================================================

class CudaPixelPipeline:
    def __init__(
        self,
        entries: Sequence[ReferenceEntry],
        width: int,
        height: int,
    ):
        self.device = torch.device(
            f"cuda:{GPU_DEVICE_INDEX}"
        )
        torch.cuda.set_device(self.device)

        if GPU_USE_TF32:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

        self.width = width
        self.height = height
        self.entries = list(entries)

        self._grid = self._create_destination_grid(
            width,
            height,
        )

        self._sobel_x = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0],
            ],
            device=self.device,
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 8.0

        self._sobel_y = self._sobel_x.transpose(2, 3).contiguous()

        self._gaussian_kernel = self._make_gaussian_kernel(
            size=11,
            sigma=5.0,
        )

        self._upload_references()

        props = torch.cuda.get_device_properties(self.device)
        print("=" * 80)
        print("PyTorch CUDA 像素流水线已启动")
        print(f"GPU：{props.name}")
        print(f"PyTorch：{torch.__version__}")
        print(f"PyTorch CUDA：{torch.version.cuda}")
        print(
            f"显存：{props.total_memory / (1024 ** 3):.2f} GB"
        )
        print("=" * 80)

    def _create_destination_grid(
        self,
        width: int,
        height: int,
    ) -> torch.Tensor:
        ys, xs = torch.meshgrid(
            torch.arange(
                height,
                device=self.device,
                dtype=torch.float32,
            ),
            torch.arange(
                width,
                device=self.device,
                dtype=torch.float32,
            ),
            indexing="ij",
        )

        ones = torch.ones_like(xs)

        return torch.stack(
            [xs, ys, ones],
            dim=-1,
        ).reshape(-1, 3).T.contiguous()

    def _make_gaussian_kernel(
        self,
        size: int,
        sigma: float,
    ) -> torch.Tensor:
        coords = torch.arange(
            size,
            device=self.device,
            dtype=torch.float32,
        ) - (size - 1) / 2.0

        kernel_1d = torch.exp(
            -(coords * coords) / (2.0 * sigma * sigma)
        )
        kernel_1d = kernel_1d / kernel_1d.sum()

        kernel_2d = torch.outer(kernel_1d, kernel_1d)

        return kernel_2d.view(1, 1, size, size)

    def _image_tensor(self, image: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        value = torch.from_numpy(
            np.ascontiguousarray(rgb)
        ).to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        return value.permute(2, 0, 1).unsqueeze(0) / 255.0

    def _mask_tensor(self, mask: np.ndarray) -> torch.Tensor:
        value = torch.from_numpy(
            np.ascontiguousarray(mask)
        ).to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        return value.unsqueeze(0).unsqueeze(0) / 255.0

    def _upload_references(self) -> None:
        with torch.inference_mode():
            for entry in self.entries:
                entry.gpu_image = self._image_tensor(entry.image)
                entry.gpu_road_mask = self._mask_tensor(entry.road_mask)
                entry.gpu_vehicle_mask = self._mask_tensor(
                    entry.vehicle_mask
                )
        torch.cuda.synchronize(self.device)

    def _warp_batch(
        self,
        source: torch.Tensor,
        homographies: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        """
        homographies：source 像素坐标 -> destination 像素坐标。
        grid_sample 需要 destination -> source，所以内部取逆矩阵。
        """
        batch = source.shape[0]
        inverse = torch.linalg.inv(homographies)

        grid = self._grid.unsqueeze(0).expand(batch, -1, -1)
        source_points = torch.bmm(
            inverse,
            grid,
        )

        denominator = source_points[:, 2:3, :]
        denominator = torch.where(
            denominator.abs() < 1e-7,
            torch.full_like(denominator, 1e-7),
            denominator,
        )

        x = source_points[:, 0:1, :] / denominator
        y = source_points[:, 1:2, :] / denominator

        x_normalized = (
            2.0 * x / max(self.width - 1, 1) - 1.0
        )
        y_normalized = (
            2.0 * y / max(self.height - 1, 1) - 1.0
        )

        sample_grid = torch.cat(
            [x_normalized, y_normalized],
            dim=1,
        )
        sample_grid = sample_grid.permute(0, 2, 1).reshape(
            batch,
            self.height,
            self.width,
            2,
        )

        return F.grid_sample(
            source,
            sample_grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=True,
        )

    @staticmethod
    def _dilate(value: torch.Tensor, kernel_size: int) -> torch.Tensor:
        if kernel_size <= 1:
            return value
        return F.max_pool2d(
            value,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    @staticmethod
    def _erode(value: torch.Tensor, kernel_size: int) -> torch.Tensor:
        if kernel_size <= 1:
            return value
        return -F.max_pool2d(
            -value,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    def _open_close(self, value: torch.Tensor) -> torch.Tensor:
        opened = self._dilate(
            self._erode(value, OPEN_KERNEL_SIZE),
            OPEN_KERNEL_SIZE,
        )
        closed = self._erode(
            self._dilate(opened, CLOSE_KERNEL_SIZE),
            CLOSE_KERNEL_SIZE,
        )
        return closed

    def _gaussian_blur(self, value: torch.Tensor) -> torch.Tensor:
        channels = value.shape[1]
        kernel = self._gaussian_kernel.expand(
            channels,
            1,
            -1,
            -1,
        )
        return F.conv2d(
            value,
            kernel,
            padding=self._gaussian_kernel.shape[-1] // 2,
            groups=channels,
        )

    def _sobel(self, gray: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(gray, self._sobel_x, padding=1)
        gy = F.conv2d(gray, self._sobel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)

    @staticmethod
    def _gray(rgb: torch.Tensor) -> torch.Tensor:
        return (
            rgb[:, 0:1] * 0.299
            + rgb[:, 1:2] * 0.587
            + rgb[:, 2:3] * 0.114
        )

    def process(
        self,
        current_frame: np.ndarray,
        current_road_mask: np.ndarray,
        current_vehicle_mask: np.ndarray,
        aligned_references: Sequence[AlignedReference],
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        List[float],
        float,
    ]:
        if len(aligned_references) < MIN_REFERENCE_VOTES:
            empty_u8 = np.zeros(
                (self.height, self.width),
                dtype=np.uint8,
            )
            empty_f32 = np.zeros(
                (self.height, self.width),
                dtype=np.float32,
            )
            return empty_u8, empty_f32, [], 0.0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.inference_mode():
            start_event.record()

            current = self._image_tensor(current_frame)
            current_road = self._mask_tensor(current_road_mask)
            current_vehicle = self._mask_tensor(current_vehicle_mask)

            reference_images = torch.cat(
                [
                    item.entry.gpu_image
                    for item in aligned_references
                    if item.entry.gpu_image is not None
                ],
                dim=0,
            )
            reference_roads = torch.cat(
                [
                    item.entry.gpu_road_mask
                    for item in aligned_references
                    if item.entry.gpu_road_mask is not None
                ],
                dim=0,
            )
            reference_vehicles = torch.cat(
                [
                    item.entry.gpu_vehicle_mask
                    for item in aligned_references
                    if item.entry.gpu_vehicle_mask is not None
                ],
                dim=0,
            )

            homographies = torch.as_tensor(
                np.stack(
                    [item.homography for item in aligned_references],
                    axis=0,
                ),
                device=self.device,
                dtype=torch.float32,
            )

            warped_references = self._warp_batch(
                reference_images,
                homographies,
                mode="bilinear",
            )
            warped_roads = self._warp_batch(
                reference_roads,
                homographies,
                mode="nearest",
            )
            warped_reference_vehicles = self._warp_batch(
                reference_vehicles,
                homographies,
                mode="nearest",
            )

            batch = warped_references.shape[0]
            current_batch = current.expand(batch, -1, -1, -1)
            current_road_batch = current_road.expand(
                batch,
                -1,
                -1,
                -1,
            )
            current_vehicle_batch = current_vehicle.expand(
                batch,
                -1,
                -1,
                -1,
            )

            current_vehicle_expanded = self._dilate(
                current_vehicle_batch,
                VEHICLE_EXTRA_DILATE,
            )
            reference_vehicle_expanded = self._dilate(
                warped_reference_vehicles,
                VEHICLE_EXTRA_DILATE,
            )

            valid = (
                (current_road_batch > 0.5)
                & (warped_roads > 0.5)
                & (current_vehicle_expanded <= 0.5)
                & (reference_vehicle_expanded <= 0.5)
            ).float()

            valid = self._erode(
                valid,
                ROAD_BOUNDARY_ERODE,
            )
            valid = (valid > 0.5).float()

            current_gray = self._gray(current_batch)
            reference_gray = self._gray(warped_references)

            valid_count = valid.sum(
                dim=(2, 3),
                keepdim=True,
            ).clamp_min(100.0)

            current_mean = (
                current_gray * valid
            ).sum(dim=(2, 3), keepdim=True) / valid_count

            reference_mean = (
                reference_gray * valid
            ).sum(dim=(2, 3), keepdim=True) / valid_count

            current_variance = (
                (current_gray - current_mean).pow(2) * valid
            ).sum(dim=(2, 3), keepdim=True) / valid_count

            reference_variance = (
                (reference_gray - reference_mean).pow(2) * valid
            ).sum(dim=(2, 3), keepdim=True) / valid_count

            current_std = current_variance.sqrt().clamp_min(0.025)
            reference_std = reference_variance.sqrt().clamp_min(0.025)

            adjusted_reference_gray = (
                (reference_gray - reference_mean)
                * (current_std / reference_std)
                + current_mean
            ).clamp(0.0, 1.0)

            luminance_diff = (
                current_gray - adjusted_reference_gray
            ).abs() * 255.0

            current_chroma = current_batch - current_gray
            reference_chroma = warped_references - reference_gray

            color_diff = (
                current_chroma - reference_chroma
            ).abs().mean(dim=1, keepdim=True) * 255.0

            current_blur = self._gaussian_blur(current_gray)
            reference_blur = self._gaussian_blur(
                adjusted_reference_gray
            )

            current_detail = (current_gray - current_blur).abs()
            reference_detail = (
                adjusted_reference_gray - reference_blur
            ).abs()

            texture_diff = (
                current_detail - reference_detail
            ).abs() * 255.0

            current_edge = self._sobel(current_gray)
            reference_edge = self._sobel(adjusted_reference_gray)
            edge_diff = (
                current_edge - reference_edge
            ).abs().clamp(0.0, 1.0) * 255.0

            score = (
                0.50 * luminance_diff
                + 0.20 * color_diff
                + 0.18 * texture_diff
                + 0.12 * edge_diff
            )

            score = F.avg_pool2d(
                score,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            score = score * valid

            sampled_score = score[
                :,
                :,
                ::THRESHOLD_SAMPLE_STRIDE,
                ::THRESHOLD_SAMPLE_STRIDE,
            ]
            sampled_valid = valid[
                :,
                :,
                ::THRESHOLD_SAMPLE_STRIDE,
                ::THRESHOLD_SAMPLE_STRIDE,
            ] > 0.5

            thresholds = []

            for index in range(batch):
                values = sampled_score[index][sampled_valid[index]]

                if values.numel() < 100:
                    threshold = torch.tensor(
                        BASE_DIFF_THRESHOLD,
                        device=self.device,
                        dtype=torch.float32,
                    )
                else:
                    median = values.median()
                    mad = (values - median).abs().median()
                    threshold = (
                        median
                        + MAD_MULTIPLIER * mad
                        + MAD_EXTRA
                    )
                    threshold = threshold.clamp(
                        min=BASE_DIFF_THRESHOLD,
                        max=MAX_ADAPTIVE_THRESHOLD,
                    )

                thresholds.append(threshold)

            threshold_tensor = torch.stack(
                thresholds,
                dim=0,
            ).view(batch, 1, 1, 1)

            binary = (
                (score >= threshold_tensor)
                & (valid > 0.5)
            )

            vote_map = binary.sum(dim=0)
            valid_union = (valid > 0.5).any(dim=0)

            voted = (
                (vote_map >= MIN_REFERENCE_VOTES)
                & valid_union
            ).float().unsqueeze(0)

            voted = self._open_close(voted)
            voted = (voted > 0.5).float()

            combined_score = score.mean(dim=0)

            end_event.record()
            torch.cuda.synchronize(self.device)

            gpu_ms = float(start_event.elapsed_time(end_event))

            voted_cpu = (
                voted[0, 0]
                .mul(255.0)
                .clamp(0.0, 255.0)
                .byte()
                .cpu()
                .numpy()
            )

            score_cpu = (
                combined_score[0]
                .float()
                .cpu()
                .numpy()
            )

            threshold_values = [
                float(item.detach().cpu())
                for item in thresholds
            ]

        return (
            voted_cpu,
            score_cpu,
            threshold_values,
            gpu_ms,
        )


# ============================================================
# 候选区域
# ============================================================

def perspective_min_area(
    center_y: float,
    image_height: int,
) -> int:
    ratio = np.clip(
        center_y / max(image_height - 1, 1),
        0.0,
        1.0,
    )
    return int(
        round(
            TOP_MIN_AREA
            + (BOTTOM_MIN_AREA - TOP_MIN_AREA)
            * ratio
            * ratio
        )
    )


def extract_candidate_regions(
    binary_mask: np.ndarray,
    combined_score: np.ndarray,
    road_mask: np.ndarray,
) -> List[CandidateRegion]:
    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8,
        )
    )

    road_pixels = max(
        int(np.count_nonzero(road_mask)),
        1,
    )

    candidates: List[CandidateRegion] = []

    for label_id in range(1, count):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        center_y = float(centroids[label_id][1])

        if area < perspective_min_area(center_y, binary_mask.shape[0]):
            continue

        if area / float(road_pixels) > MAX_COMPONENT_ROAD_RATIO:
            continue

        if width < 6 or height < 6:
            continue

        component = labels == label_id
        mean_score = float(combined_score[component].mean())

        if mean_score < MIN_COMPONENT_MEAN_SCORE:
            continue

        candidates.append(
            CandidateRegion(
                bbox=(x, y, x + width, y + height),
                area=area,
                mean_score=mean_score,
                vote_mean=float(MIN_REFERENCE_VOTES),
            )
        )

    return candidates


# ============================================================
# 时间跟踪
# ============================================================

def bbox_iou(
    first: Tuple[int, int, int, int],
    second: Tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = (
        max(0, ix2 - ix1)
        * max(0, iy2 - iy1)
    )

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))

    return intersection / max(
        area_a + area_b - intersection,
        1,
    )


def bbox_center_distance(
    first: Tuple[int, int, int, int],
    second: Tuple[int, int, int, int],
) -> Tuple[float, float]:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second

    center_a = np.asarray(
        [(ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0],
        dtype=np.float32,
    )
    center_b = np.asarray(
        [(bx1 + bx2) / 2.0, (by1 + by2) / 2.0],
        dtype=np.float32,
    )

    distance = float(np.linalg.norm(center_a - center_b))
    scale = max(
        math.hypot(ax2 - ax1, ay2 - ay1),
        math.hypot(bx2 - bx1, by2 - by1),
        1.0,
    )
    return distance, scale


class TemporalAnomalyTracker:
    def __init__(self):
        self.tracks: List[AnomalyTrack] = []
        self.next_track_id = 1

    def update(
        self,
        candidates: List[CandidateRegion],
        timestamp: float,
    ) -> None:
        for track in self.tracks:
            track.matched_this_update = False

        pairs = []

        for track_index, track in enumerate(self.tracks):
            for candidate_index, candidate in enumerate(candidates):
                iou = bbox_iou(track.bbox, candidate.bbox)
                distance, scale = bbox_center_distance(
                    track.bbox,
                    candidate.bbox,
                )

                if (
                    iou >= TRACK_IOU_THRESHOLD
                    or distance
                    <= scale * TRACK_CENTER_DISTANCE_RATIO
                ):
                    pairs.append(
                        (
                            (1.0 - iou) * 100.0 + distance,
                            track_index,
                            candidate_index,
                        )
                    )

        pairs.sort(key=lambda item: item[0])

        used_tracks = set()
        used_candidates = set()

        for _, track_index, candidate_index in pairs:
            if (
                track_index in used_tracks
                or candidate_index in used_candidates
            ):
                continue

            track = self.tracks[track_index]
            candidate = candidates[candidate_index]

            old_box = np.asarray(track.bbox, dtype=np.float32)
            new_box = np.asarray(candidate.bbox, dtype=np.float32)
            smoothed = 0.58 * old_box + 0.42 * new_box

            track.bbox = tuple(
                int(round(value))
                for value in smoothed
            )
            track.last_seen = timestamp
            track.hits += 1
            track.misses = 0
            track.mean_score = (
                0.70 * track.mean_score
                + 0.30 * candidate.mean_score
            )
            track.vote_mean = (
                0.70 * track.vote_mean
                + 0.30 * candidate.vote_mean
            )
            track.matched_this_update = True

            if (
                track.hits >= CONFIRM_MIN_HITS
                and track.duration >= CONFIRM_MIN_SECONDS
            ):
                track.confirmed = True

            used_tracks.add(track_index)
            used_candidates.add(candidate_index)

        for index, track in enumerate(self.tracks):
            if index not in used_tracks:
                track.misses += 1

        for index, candidate in enumerate(candidates):
            if index in used_candidates:
                continue

            self.tracks.append(
                AnomalyTrack(
                    track_id=self.next_track_id,
                    bbox=candidate.bbox,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    hits=1,
                    mean_score=candidate.mean_score,
                    vote_mean=candidate.vote_mean,
                )
            )
            self.next_track_id += 1

        self.tracks = [
            track
            for track in self.tracks
            if track.misses <= MAX_TRACK_MISSES
        ]

    def active_tracks(self) -> List[AnomalyTrack]:
        return list(self.tracks)


# ============================================================
# NVENC / 视频写入
# ============================================================

class OpenCvVideoWriter:
    def __init__(
        self,
        path: Path,
        fps: float,
        size: Tuple[int, int],
    ):
        self.backend_name = "OpenCV mp4v (CPU)"
        self.writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            size,
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"无法创建输出视频：{path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()


class NvencVideoWriter:
    def __init__(
        self,
        ffmpeg_path: str,
        output_path: Path,
        fps: float,
        size: Tuple[int, int],
    ):
        width, height = size
        self.backend_name = "FFmpeg h264_nvenc (GPU)"
        self.output_path = output_path

        command = [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "h264_nvenc",
            "-preset",
            NVENC_PRESET,
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            str(NVENC_CQ),
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]

        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(
                subprocess,
                "CREATE_NO_WINDOW",
                0,
            )

        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )

        if self.process.stdin is None:
            raise RuntimeError("FFmpeg stdin 创建失败")

    def write(self, frame: np.ndarray) -> None:
        if self.process.poll() is not None:
            stderr = b""
            if self.process.stderr is not None:
                stderr = self.process.stderr.read()
            raise RuntimeError(
                "FFmpeg NVENC 已异常退出："
                + stderr.decode("utf-8", errors="replace")
            )

        contiguous = np.ascontiguousarray(frame, dtype=np.uint8)
        try:
            self.process.stdin.write(contiguous.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError("FFmpeg NVENC 管道已断开") from exc

    def close(self) -> None:
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except Exception:
                pass

        try:
            return_code = self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            return_code = self.process.wait()

        if return_code != 0:
            stderr = b""
            if self.process.stderr is not None:
                stderr = self.process.stderr.read()
            print(
                "[NVENC] FFmpeg 退出异常："
                + stderr.decode("utf-8", errors="replace")
            )


def ffmpeg_supports_nvenc() -> Tuple[Optional[str], str]:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return None, "PATH 中没有 ffmpeg"

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(
            subprocess,
            "CREATE_NO_WINDOW",
            0,
        )

    try:
        result = subprocess.run(
            [
                ffmpeg_path,
                "-hide_banner",
                "-encoders",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            creationflags=creationflags,
        )
    except Exception as exc:
        return None, f"ffmpeg 检查失败：{exc}"

    content = (result.stdout or "") + (result.stderr or "")

    if "h264_nvenc" not in content:
        return None, "当前 ffmpeg 不包含 h264_nvenc"

    return ffmpeg_path, "ok"


def create_video_writer(
    output_path: Path,
    fps: float,
    size: Tuple[int, int],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if PREFER_NVENC:
        ffmpeg_path, reason = ffmpeg_supports_nvenc()

        if ffmpeg_path:
            try:
                writer = NvencVideoWriter(
                    ffmpeg_path,
                    output_path,
                    fps,
                    size,
                )
                print("[编码] 使用 FFmpeg h264_nvenc")
                return writer
            except Exception as exc:
                reason = f"NVENC 启动失败：{exc}"

        if REQUIRE_NVENC:
            raise RuntimeError(reason)

        print(f"[编码] NVENC 不可用，回退 CPU：{reason}")

    return OpenCvVideoWriter(
        output_path,
        fps,
        size,
    )


# ============================================================
# 显示
# ============================================================

def create_debug_inset(
    voted_mask: np.ndarray,
    score: np.ndarray,
    vehicle_mask: np.ndarray,
    width: int,
) -> np.ndarray:
    height, source_width = voted_mask.shape[:2]

    normalized = np.clip(score, 0, 100)
    normalized = (normalized / 100.0 * 255.0).astype(np.uint8)

    panel = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET,
    )
    panel[voted_mask > 0] = (0, 0, 255)
    panel[vehicle_mask > 0] = (255, 80, 20)

    scale = min(1.0, width / max(source_width, 1))
    panel = cv2.resize(
        panel,
        (
            max(1, int(round(source_width * scale))),
            max(1, int(round(height * scale))),
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    cv2.rectangle(
        panel,
        (0, 0),
        (panel.shape[1] - 1, panel.shape[0] - 1),
        (245, 245, 245),
        2,
    )
    cv2.putText(
        panel,
        "GPU difference / vote mask",
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
    selected_references: Sequence[AlignedReference],
    thresholds: List[float],
    inference_ms: float,
    gpu_pipeline_ms: float,
    processing_fps: float,
    frame_index: int,
    total_frames: int,
    paused: bool,
    encoder_name: str,
) -> np.ndarray:
    output = frame.copy()

    road_overlay = output.copy()
    road_overlay[road_mask > 0] = (45, 150, 90)
    output = cv2.addWeighted(
        road_overlay,
        0.10,
        output,
        0.90,
        0,
    )

    polygon = np.round(
        road_polygon
    ).astype(np.int32).reshape(-1, 1, 2)

    cv2.polylines(
        output,
        [polygon],
        True,
        (0, 225, 255),
        4,
        cv2.LINE_AA,
    )

    vehicle_overlay = output.copy()
    vehicle_overlay[vehicle_mask > 0] = (35, 35, 230)
    output = cv2.addWeighted(
        vehicle_overlay,
        0.25,
        output,
        0.75,
        0,
    )

    for item in accepted_detections:
        detection = item[0]
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (40, 220, 255),
            2,
            cv2.LINE_AA,
        )

    confirmed_count = 0

    for track in tracks:
        x1, y1, x2, y2 = track.bbox

        if track.confirmed:
            confirmed_count += 1
            color = (20, 20, 245)
            prefix = "ANOMALY"
            thickness = 4
        else:
            color = (0, 165, 255)
            prefix = "candidate"
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
            f"{prefix} #{track.track_id} "
            f"{track.duration:.1f}s "
            f"score={track.mean_score:.0f}"
        )
        label_y = max(24, y1 - 8)

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
            int(round(output.shape[1] * 0.28)),
        ),
    )

    inset_x = output.shape[1] - inset.shape[1] - 15
    inset_y = 116

    if (
        inset_x >= 0
        and inset_y + inset.shape[0] <= output.shape[0]
    ):
        output[
            inset_y : inset_y + inset.shape[0],
            inset_x : inset_x + inset.shape[1],
        ] = inset

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 108),
        (9, 16, 25),
        -1,
    )

    state = "PAUSED" if paused else "PLAYING"

    reference_names = ",".join(
        item.entry.name.replace("reference_", "R").replace(".png", "")
        for item in selected_references
    ) or "none"

    threshold_text = (
        ",".join(f"{value:.0f}" for value in thresholds)
        if thresholds
        else "-"
    )

    candidate_count = sum(
        1 for track in tracks if not track.confirmed
    )

    lines = [
        (
            f"{state} | Frame {frame_index}/"
            f"{total_frames if total_frames > 0 else '?'} "
            f"| FPS {processing_fps:.1f}",
            (245, 245, 245),
        ),
        (
            f"YOLO CUDA {inference_ms:.1f}ms | "
            f"GPU diff {gpu_pipeline_ms:.1f}ms | "
            f"refs {reference_names} | th {threshold_text}",
            (80, 230, 255),
        ),
        (
            f"vehicles {len(accepted_detections)} | "
            f"candidates {candidate_count} | "
            f"confirmed {confirmed_count} | "
            f"encoder {encoder_name}",
            (120, 240, 150),
        ),
    ]

    for index, (text, color) in enumerate(lines):
        cv2.putText(
            output,
            text,
            (16, 28 + index * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.61,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    project_root = Path(__file__).resolve().parent

    input_name = (
        sys.argv[1]
        if len(sys.argv) >= 2
        else DEFAULT_VIDEO_FILENAME
    )

    input_path = (
        Path(input_name)
        if Path(input_name).is_absolute()
        else project_root / input_name
    )

    model_path = project_root / MODEL_FILENAME
    anchor_image_path = project_root / ANCHOR_IMAGE_PATH
    roi_json_path = project_root / ANCHOR_ROI_JSON_PATH
    anchor_mask_path = project_root / ANCHOR_ROAD_MASK_PATH
    output_path = project_root / OUTPUT_VIDEO_PATH

    for path, label in (
        (input_path, "输入视频"),
        (model_path, "normal.onnx"),
        (anchor_image_path, "reference_01.png"),
        (roi_json_path, "road_roi.json"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"没有找到{label}：{path}")

    anchor_image = imread_unicode(
        anchor_image_path,
        cv2.IMREAD_COLOR,
    )
    if anchor_image is None:
        raise RuntimeError(f"无法读取主基准图：{anchor_image_path}")

    anchor_height, anchor_width = anchor_image.shape[:2]
    anchor_polygon = load_roi_points(roi_json_path)

    anchor_road_mask = None
    if anchor_mask_path.exists():
        anchor_road_mask = imread_unicode(
            anchor_mask_path,
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
        (anchor_width, anchor_height),
        interpolation=cv2.INTER_NEAREST,
    )

    detector = NormalOnnxVehicleDetector(model_path)

    actual_providers = list(detector.session.get_providers())
    print(f"[CUDA] normal.onnx 实际 Provider：{actual_providers}")

    if (
        not actual_providers
        or actual_providers[0] != "CUDAExecutionProvider"
    ):
        raise RuntimeError(
            "normal.onnx 没有成功使用 GPU："
            f"{actual_providers}"
        )

    road_tracker = RoadRoiTracker(
        anchor_image=anchor_image,
        anchor_polygon=anchor_polygon,
        anchor_road_mask=anchor_road_mask,
    )

    reference_bank = ReferenceBank(
        project_root=project_root,
        detector=detector,
        anchor_image=anchor_image,
        anchor_size=(anchor_width, anchor_height),
    )

    gpu_pipeline = CudaPixelPipeline(
        entries=reference_bank.entries,
        width=anchor_width,
        height=anchor_height,
    )

    capture, temp_video_path = open_video_with_fallback(input_path)
    writer = None

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(source_fps) or source_fps <= 1.0:
            source_fps = 25.0

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if SAVE_OUTPUT_VIDEO:
            writer = create_video_writer(
                output_path,
                source_fps,
                (anchor_width, anchor_height),
            )

        encoder_name = (
            writer.backend_name
            if writer is not None
            else "disabled"
        )

        print("=" * 80)
        print("多基准道路障碍物检测 · GPU 像素流水线")
        print(f"输入视频：{input_path}")
        print(f"可用基准：{len(reference_bank.entries)}")
        print(f"编码器：{encoder_name}")
        print(f"输出视频：{output_path}")
        print("=" * 80)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        anomaly_tracker = TemporalAnomalyTracker()

        paused = False
        single_step = False
        frame_index = 0

        last_frame: Optional[np.ndarray] = None
        last_road_polygon = anchor_polygon.copy()
        last_anchor_to_current = np.eye(3, dtype=np.float64)

        last_detections = []
        last_inference_ms = 0.0
        last_vehicle_mask = np.zeros(
            (anchor_height, anchor_width),
            dtype=np.uint8,
        )
        last_accepted = []

        last_selected: List[AlignedReference] = []
        last_voted_mask = np.zeros(
            (anchor_height, anchor_width),
            dtype=np.uint8,
        )
        last_score = np.zeros(
            (anchor_height, anchor_width),
            dtype=np.float32,
        )
        last_thresholds: List[float] = []
        last_gpu_ms = 0.0

        fps_ema = 0.0

        while True:
            should_read = (
                not paused
                or single_step
                or last_frame is None
            )

            if should_read:
                success, source_frame = capture.read()
                if not success or source_frame is None:
                    print("视频播放结束。")
                    break

                frame_index += 1
                frame = cv2.resize(
                    source_frame,
                    (anchor_width, anchor_height),
                    interpolation=cv2.INTER_AREA,
                )
                last_frame = frame
                loop_started = time.perf_counter()

                if (
                    frame_index == 1
                    or frame_index % ROAD_ROI_UPDATE_EVERY_N_FRAMES == 0
                ):
                    (
                        last_road_polygon,
                        _,
                        _,
                    ) = road_tracker.update(frame)

                    last_anchor_to_current = as_homography(
                        road_tracker.last_homography
                    )

                road_mask = polygon_to_mask(
                    last_road_polygon,
                    anchor_width,
                    anchor_height,
                )

                if (
                    frame_index == 1
                    or frame_index % YOLO_EVERY_N_FRAMES == 0
                ):
                    (
                        last_detections,
                        last_inference_ms,
                    ) = detector.detect(frame)

                (
                    last_vehicle_mask,
                    last_accepted,
                ) = build_vehicle_mask(
                    last_detections,
                    detector,
                    road_mask,
                )

                if (
                    frame_index == 1
                    or frame_index % ANOMALY_EVERY_N_FRAMES == 0
                ):
                    last_selected = reference_bank.select(
                        frame,
                        last_anchor_to_current,
                    )

                    (
                        last_voted_mask,
                        last_score,
                        last_thresholds,
                        last_gpu_ms,
                    ) = gpu_pipeline.process(
                        current_frame=frame,
                        current_road_mask=road_mask,
                        current_vehicle_mask=last_vehicle_mask,
                        aligned_references=last_selected,
                    )

                    candidates = extract_candidate_regions(
                        last_voted_mask,
                        last_score,
                        road_mask,
                    )

                    anomaly_tracker.update(
                        candidates,
                        frame_index / source_fps,
                    )

                elapsed = time.perf_counter() - loop_started
                instant_fps = 1.0 / elapsed if elapsed > 0 else 0.0

                if fps_ema <= 0:
                    fps_ema = instant_fps
                else:
                    fps_ema = 0.90 * fps_ema + 0.10 * instant_fps

                result = draw_result(
                    frame=frame,
                    road_polygon=last_road_polygon,
                    road_mask=road_mask,
                    vehicle_mask=last_vehicle_mask,
                    accepted_detections=last_accepted,
                    tracks=anomaly_tracker.active_tracks(),
                    voted_mask=last_voted_mask,
                    combined_score=last_score,
                    selected_references=last_selected,
                    thresholds=last_thresholds,
                    inference_ms=last_inference_ms,
                    gpu_pipeline_ms=last_gpu_ms,
                    processing_fps=fps_ema,
                    frame_index=frame_index,
                    total_frames=total_frames,
                    paused=paused,
                    encoder_name=encoder_name,
                )

                if writer is not None:
                    writer.write(result)

                single_step = False

            else:
                road_mask = polygon_to_mask(
                    last_road_polygon,
                    anchor_width,
                    anchor_height,
                )

                result = draw_result(
                    frame=last_frame,
                    road_polygon=last_road_polygon,
                    road_mask=road_mask,
                    vehicle_mask=last_vehicle_mask,
                    accepted_detections=last_accepted,
                    tracks=anomaly_tracker.active_tracks(),
                    voted_mask=last_voted_mask,
                    combined_score=last_score,
                    selected_references=last_selected,
                    thresholds=last_thresholds,
                    inference_ms=last_inference_ms,
                    gpu_pipeline_ms=last_gpu_ms,
                    processing_fps=fps_ema,
                    frame_index=frame_index,
                    total_frames=total_frames,
                    paused=True,
                    encoder_name=encoder_name,
                )

            display = resize_to_fit(
                result,
                MAX_DISPLAY_WIDTH,
                MAX_DISPLAY_HEIGHT,
            )
            cv2.imshow(WINDOW_NAME, display)

            delay = (
                max(1, int(round(1000.0 / source_fps)))
                if not paused
                else 30
            )
            key = cv2.waitKey(delay)
            key_code = -1 if key == -1 else key & 0xFF

            if key_code in (ord("q"), ord("Q"), 27):
                break

            if key_code == ord(" "):
                paused = not paused
            elif key_code in (ord("s"), ord("S")) and paused:
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
            writer.close()

        cv2.destroyAllWindows()

        if temp_video_path is not None:
            temp_video_path.unlink(missing_ok=True)

    print("=" * 80)
    print("处理结束")
    print(f"结果视频：{output_path}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        sys.exit(0)
    except Exception as exc:
        cv2.destroyAllWindows()
        print(f"[失败] {type(exc).__name__}: {exc}")
        raise

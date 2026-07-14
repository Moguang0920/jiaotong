# -*- coding: utf-8 -*-
"""
道路异常检测链路 · 第一步完善版

功能：
1. 读取项目根目录“正常道路.mp4”。
2. 读取已经在 reference_01.png 上标定的道路 ROI。
3. 使用 ORB + Homography，让道路 ROI 跟随视频轻微晃动和透视变化。
4. 加载项目根目录 normal.onnx，优先使用 CUDAExecutionProvider。
5. 在当前帧中检测车辆。
6. 只保留道路 ROI 内的车辆，并扩大车辆框生成“车辆忽略 Mask”。
7. 生成：
       有效道路区域 = 当前道路 ROI - 当前车辆 Mask
8. 实时显示道路框、车辆框、车辆忽略区域、最终有效检测区域。
9. 保存可视化视频。

本阶段不进行道路障碍物差分判断。
下一阶段才会把“有效道路区域”与多张正常道路基准图比较。

项目目录要求：
Jiaotong-gpt/
├── 正常道路.mp4
├── normal.onnx
├── road_roi_vehicle_mask_demo.py
└── road_anomaly_data/
    └── camera_01/
        ├── reference_bank/
        │   └── reference_01.png
        └── road_roi/
            ├── road_roi.json
            └── road_mask.png

运行：
    python road_roi_vehicle_mask_demo.py

操作：
- Q / Esc：退出
- Space：暂停 / 继续
- S：暂停时单步前进
"""

from __future__ import annotations

import ast
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception as exc:
    ort = None
    ORT_IMPORT_ERROR = exc
else:
    ORT_IMPORT_ERROR = None


# ============================================================
# 路径
# ============================================================

VIDEO_FILENAME = "正常道路.mp4"
MODEL_FILENAME = "normal.onnx"

CAMERA_ROOT = Path("road_anomaly_data") / "camera_01"
ANCHOR_IMAGE_PATH = (
    CAMERA_ROOT / "reference_bank" / "reference_01.png"
)
ROAD_ROI_JSON_PATH = (
    CAMERA_ROOT / "road_roi" / "road_roi.json"
)
ANCHOR_ROAD_MASK_PATH = (
    CAMERA_ROOT / "road_roi" / "road_mask.png"
)

OUTPUT_VIDEO_PATH = (
    CAMERA_ROOT / "road_roi_vehicle_mask_overlay.mp4"
)


# ============================================================
# 运行参数
# ============================================================

WINDOW_NAME = "Road ROI + normal.onnx Vehicle Mask"

SAVE_OUTPUT_VIDEO = True
MAX_DISPLAY_WIDTH = 1700
MAX_DISPLAY_HEIGHT = 940

# YOLO 参数
YOLO_CONFIDENCE = 0.35
YOLO_NMS_IOU = 0.45

# 1 表示每帧检测；如果速度不足可以改成 2。
YOLO_EVERY_N_FRAMES = 1

# normal.onnx 专门用于检测车辆，因此默认把模型输出的所有类别
# 都视为“需要排除的车辆类别”。
#
# 如果你的 normal.onnx 后续包含别的类别，可改成 False，
# 再通过 FORCE_VEHICLE_CLASS_IDS 指定车辆类别编号。
TREAT_ALL_DETECTIONS_AS_VEHICLES = True

# 示例：只把类别0视为车辆：
# FORCE_VEHICLE_CLASS_IDS: Optional[Set[int]] = {0}
FORCE_VEHICLE_CLASS_IDS: Optional[Set[int]] = None

# 车辆框向外扩张，避免车身边缘、阴影、检测框抖动进入差分区域。
VEHICLE_MASK_EXPAND_X = 0.12
VEHICLE_MASK_EXPAND_TOP = 0.08
VEHICLE_MASK_EXPAND_BOTTOM = 0.15

# 车辆框与道路区域相交比例达到该值，或者车辆框底部中心位于道路内，
# 才认为是道路上的车辆。
MIN_BOX_ROAD_INTERSECTION = 0.12

# 过小的框直接丢弃。
MIN_BOX_WIDTH = 6
MIN_BOX_HEIGHT = 6


# ============================================================
# ROI 跟随参数
# ============================================================

# ORB/Homography 不必每帧重算。
# 2 表示每两帧更新一次，其他帧使用上一次结果。
ROI_UPDATE_EVERY_N_FRAMES = 2

ORB_FEATURES = 7000
ORB_LOWE_RATIO = 0.76
MIN_GOOD_MATCHES = 24
MIN_HOMOGRAPHY_INLIERS = 16
MIN_HOMOGRAPHY_INLIER_RATIO = 0.28

# 道路框坐标平滑。
# 越大越跟手，越小越稳定。
ROI_POLYGON_EMA_ALPHA = 0.42

# 合理晃动范围保护，避免错误匹配把道路框甩到画面外。
MAX_CENTER_SHIFT_X_RATIO = 0.25
MAX_CENTER_SHIFT_Y_RATIO = 0.25
MIN_TRANSFORMED_AREA_RATIO = 0.70
MAX_TRANSFORMED_AREA_RATIO = 1.35
MAX_PERSPECTIVE_TERM = 0.0045


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


@dataclass
class LetterboxInfo:
    ratio: float
    pad_x: float
    pad_y: float
    input_width: int
    input_height: int


# ============================================================
# 中文路径兼容
# ============================================================

def imread_unicode(
    path: Path,
    flags: int = cv2.IMREAD_COLOR,
) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def open_video_with_fallback(
    video_path: Path,
) -> Tuple[cv2.VideoCapture, Optional[Path]]:
    capture = cv2.VideoCapture(str(video_path))

    if capture.isOpened():
        return capture, None

    capture.release()

    temp_path = video_path.with_name("_normal_road_temp.mp4")
    print(
        "[提示] OpenCV 无法直接读取中文视频路径，"
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
# ROI 数据
# ============================================================

def load_roi_points(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_points = data.get("points", [])

    if len(raw_points) < 3:
        raise RuntimeError(
            f"road_roi.json 中只有 {len(raw_points)} 个点，至少需要3个。"
        )

    points = np.asarray(
        [
            [float(item["x"]), float(item["y"])]
            for item in raw_points
        ],
        dtype=np.float32,
    )

    return points


def polygon_to_mask(
    polygon: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)

    polygon_int = np.round(
        polygon
    ).astype(np.int32).reshape((-1, 1, 2))

    cv2.fillPoly(mask, [polygon_int], 255)
    return mask


def clip_polygon(
    polygon: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    result = polygon.copy()
    result[:, 0] = np.clip(result[:, 0], 0, width - 1)
    result[:, 1] = np.clip(result[:, 1], 0, height - 1)
    return result


# ============================================================
# 道路 ROI 跟随器
# ============================================================

class RoadRoiTracker:
    """
    不是机器学习道路分割。

    道路区域来自人工标定。
    ORB + Homography 只负责计算摄像头当前帧相对基准图的晃动，
    再把人工道路多边形映射到当前帧。
    """

    def __init__(
        self,
        anchor_image: np.ndarray,
        anchor_polygon: np.ndarray,
        anchor_road_mask: Optional[np.ndarray],
    ):
        self.anchor_image = anchor_image
        self.anchor_polygon = anchor_polygon.astype(np.float32)

        self.height, self.width = anchor_image.shape[:2]

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

        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        anchor_gray = self._normalize_gray(anchor_image)

        # 优先在道路外的建筑、路沿、固定设施上找特征。
        # 这样车辆经过道路时不容易干扰相机位移估计。
        feature_mask = None

        if (
            anchor_road_mask is not None
            and anchor_road_mask.shape[:2]
            == anchor_image.shape[:2]
        ):
            _, road_binary = cv2.threshold(
                anchor_road_mask,
                127,
                255,
                cv2.THRESH_BINARY,
            )

            expanded_road = cv2.dilate(
                road_binary,
                np.ones((15, 15), np.uint8),
                iterations=1,
            )

            feature_mask = cv2.bitwise_not(expanded_road)

            outside_ratio = float(
                np.count_nonzero(feature_mask)
            ) / float(feature_mask.size)

            if outside_ratio < 0.12:
                feature_mask = None

        (
            self.anchor_keypoints,
            self.anchor_descriptors,
        ) = self.orb.detectAndCompute(
            anchor_gray,
            feature_mask,
        )

        # 如果道路外固定特征太少，退回整张图。
        if (
            self.anchor_descriptors is None
            or len(self.anchor_keypoints) < MIN_GOOD_MATCHES
        ):
            (
                self.anchor_keypoints,
                self.anchor_descriptors,
            ) = self.orb.detectAndCompute(
                anchor_gray,
                None,
            )

        if (
            self.anchor_descriptors is None
            or len(self.anchor_keypoints) < 10
        ):
            raise RuntimeError(
                "基准图可用 ORB 特征过少，无法进行视频道路框跟随。"
            )

        self.matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=False,
        )

        self.last_polygon = self.anchor_polygon.copy()
        self.last_homography = np.eye(3, dtype=np.float64)
        self.last_status = "anchor"
        self.last_diagnostics: Dict[str, Any] = {}

    def _normalize_gray(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray)

    def _validate_homography(
        self,
        homography: np.ndarray,
    ) -> Tuple[bool, Dict[str, float]]:
        corners = np.asarray(
            [
                [0.0, 0.0],
                [self.width - 1.0, 0.0],
                [self.width - 1.0, self.height - 1.0],
                [0.0, self.height - 1.0],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        transformed = cv2.perspectiveTransform(
            corners,
            homography,
        ).reshape(-1, 2)

        original_area = float(
            self.width * self.height
        )

        transformed_area = abs(
            float(
                cv2.contourArea(
                    transformed.astype(np.float32)
                )
            )
        )

        area_ratio = transformed_area / max(
            original_area,
            1.0,
        )

        source_center = np.asarray(
            [self.width / 2.0, self.height / 2.0],
            dtype=np.float32,
        )

        target_center = transformed.mean(axis=0)

        shift_x_ratio = abs(
            float(target_center[0] - source_center[0])
        ) / max(float(self.width), 1.0)

        shift_y_ratio = abs(
            float(target_center[1] - source_center[1])
        ) / max(float(self.height), 1.0)

        perspective_term = max(
            abs(float(homography[2, 0])),
            abs(float(homography[2, 1])),
        )

        is_convex = bool(
            cv2.isContourConvex(
                np.round(transformed)
                .astype(np.int32)
                .reshape((-1, 1, 2))
            )
        )

        valid = (
            is_convex
            and MIN_TRANSFORMED_AREA_RATIO
            <= area_ratio
            <= MAX_TRANSFORMED_AREA_RATIO
            and shift_x_ratio
            <= MAX_CENTER_SHIFT_X_RATIO
            and shift_y_ratio
            <= MAX_CENTER_SHIFT_Y_RATIO
            and perspective_term
            <= MAX_PERSPECTIVE_TERM
        )

        return valid, {
            "area_ratio": round(area_ratio, 4),
            "shift_x_ratio": round(shift_x_ratio, 4),
            "shift_y_ratio": round(shift_y_ratio, 4),
            "perspective_term": round(
                perspective_term,
                7,
            ),
        }

    def update(
        self,
        current_frame: np.ndarray,
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        current_gray = self._normalize_gray(
            current_frame
        )

        (
            current_keypoints,
            current_descriptors,
        ) = self.orb.detectAndCompute(
            current_gray,
            None,
        )

        if (
            current_descriptors is None
            or len(current_keypoints) < 10
        ):
            self.last_status = "fallback:no_current_features"
            return (
                self.last_polygon.copy(),
                True,
                {
                    "status": self.last_status,
                    "good_matches": 0,
                    "inliers": 0,
                },
            )

        knn_matches = self.matcher.knnMatch(
            self.anchor_descriptors,
            current_descriptors,
            k=2,
        )

        good_matches = []

        for pair in knn_matches:
            if len(pair) != 2:
                continue

            first, second = pair

            if (
                first.distance
                < ORB_LOWE_RATIO * second.distance
            ):
                good_matches.append(first)

        if len(good_matches) < MIN_GOOD_MATCHES:
            self.last_status = "fallback:not_enough_matches"
            return (
                self.last_polygon.copy(),
                True,
                {
                    "status": self.last_status,
                    "good_matches": len(good_matches),
                    "inliers": 0,
                },
            )

        anchor_points = np.float32(
            [
                self.anchor_keypoints[
                    match.queryIdx
                ].pt
                for match in good_matches
            ]
        ).reshape(-1, 1, 2)

        current_points = np.float32(
            [
                current_keypoints[
                    match.trainIdx
                ].pt
                for match in good_matches
            ]
        ).reshape(-1, 1, 2)

        homography, inlier_mask = cv2.findHomography(
            anchor_points,
            current_points,
            cv2.RANSAC,
            3.5,
            maxIters=4000,
            confidence=0.995,
        )

        if homography is None or inlier_mask is None:
            self.last_status = "fallback:homography_failed"
            return (
                self.last_polygon.copy(),
                True,
                {
                    "status": self.last_status,
                    "good_matches": len(good_matches),
                    "inliers": 0,
                },
            )

        inliers = int(
            inlier_mask.reshape(-1).sum()
        )

        inlier_ratio = inliers / max(
            len(good_matches),
            1,
        )

        if (
            inliers < MIN_HOMOGRAPHY_INLIERS
            or inlier_ratio
            < MIN_HOMOGRAPHY_INLIER_RATIO
        ):
            self.last_status = "fallback:weak_geometry"
            return (
                self.last_polygon.copy(),
                True,
                {
                    "status": self.last_status,
                    "good_matches": len(good_matches),
                    "inliers": inliers,
                    "inlier_ratio": round(
                        inlier_ratio,
                        4,
                    ),
                },
            )

        valid, transform_info = (
            self._validate_homography(homography)
        )

        if not valid:
            self.last_status = "fallback:transform_rejected"
            return (
                self.last_polygon.copy(),
                True,
                {
                    "status": self.last_status,
                    "good_matches": len(good_matches),
                    "inliers": inliers,
                    "inlier_ratio": round(
                        inlier_ratio,
                        4,
                    ),
                    **transform_info,
                },
            )

        transformed_polygon = (
            cv2.perspectiveTransform(
                self.anchor_polygon.reshape(
                    -1,
                    1,
                    2,
                ),
                homography,
            ).reshape(-1, 2)
        )

        transformed_polygon = clip_polygon(
            transformed_polygon,
            self.width,
            self.height,
        )

        alpha = ROI_POLYGON_EMA_ALPHA

        smoothed_polygon = (
            (1.0 - alpha) * self.last_polygon
            + alpha * transformed_polygon
        )

        smoothed_polygon = clip_polygon(
            smoothed_polygon,
            self.width,
            self.height,
        )

        self.last_polygon = smoothed_polygon
        self.last_homography = homography
        self.last_status = "tracked"

        diagnostics = {
            "status": "tracked",
            "good_matches": len(good_matches),
            "inliers": inliers,
            "inlier_ratio": round(
                inlier_ratio,
                4,
            ),
            **transform_info,
        }

        self.last_diagnostics = diagnostics

        return (
            self.last_polygon.copy(),
            False,
            diagnostics,
        )


# ============================================================
# ONNX 类别元数据
# ============================================================

def parse_class_names_value(
    raw: Any,
) -> Dict[int, str]:
    if raw is None:
        return {}

    value: Any = raw

    if isinstance(raw, str):
        text = raw.strip()

        if not text:
            return {}

        for parser in (
            json.loads,
            ast.literal_eval,
        ):
            try:
                value = parser(text)
                break
            except Exception:
                value = raw

        if isinstance(value, str):
            parts = [
                item.strip()
                for item in value.split(",")
                if item.strip()
            ]

            return {
                index: name
                for index, name in enumerate(parts)
            }

    if isinstance(value, dict):
        result: Dict[int, str] = {}

        for key, name in value.items():
            try:
                result[int(key)] = str(name)
            except Exception:
                continue

        return result

    if isinstance(value, (list, tuple)):
        return {
            index: str(name)
            for index, name in enumerate(value)
        }

    return {}


def extract_model_class_names(
    session: "ort.InferenceSession",
) -> Dict[int, str]:
    try:
        metadata = session.get_modelmeta()
        raw_map = (
            getattr(
                metadata,
                "custom_metadata_map",
                {},
            )
            or {}
        )

        for key in (
            "names",
            "classes",
            "labels",
            "class_names",
            "categories",
        ):
            if key in raw_map:
                parsed = parse_class_names_value(
                    raw_map[key]
                )

                if parsed:
                    return parsed

        for key, value in raw_map.items():
            lowered = str(key).lower()

            if (
                "name" in lowered
                or "class" in lowered
                or "label" in lowered
            ):
                parsed = parse_class_names_value(
                    value
                )

                if parsed:
                    return parsed
    except Exception:
        pass

    return {}


def normalize_class_name(
    value: Any,
) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


VEHICLE_NAME_KEYS = {
    "car",
    "vehicle",
    "vehicles",
    "bus",
    "truck",
    "van",
    "automobile",
    "motorcycle",
    "motorbike",
    "bicycle",
    "bike",
    "车辆",
    "汽车",
    "轿车",
    "卡车",
    "公交车",
    "摩托车",
}


# ============================================================
# normal.onnx 检测器
# ============================================================

class NormalOnnxVehicleDetector:
    def __init__(
        self,
        model_path: Path,
    ):
        if ort is None:
            raise RuntimeError(
                "onnxruntime 导入失败："
                f"{ORT_IMPORT_ERROR}"
            )

        self.model_path = model_path
        self.session = self._create_session(
            model_path
        )

        self.input_info = self.session.get_inputs()[0]
        self.input_name = self.input_info.name

        self.output_names = [
            output.name
            for output in self.session.get_outputs()
        ]

        (
            self.input_height,
            self.input_width,
        ) = self._resolve_input_size()

        self.class_names = (
            extract_model_class_names(
                self.session
            )
        )

        self.providers = self.session.get_providers()

        print("=" * 78)
        print("normal.onnx 加载完成")
        print(f"模型路径：{model_path}")
        print(f"Provider：{self.providers}")
        print(
            "输入尺寸："
            f"{self.input_width} x "
            f"{self.input_height}"
        )
        print(f"输出节点：{self.output_names}")
        print(
            "类别名称："
            f"{self.class_names or '模型未写入类别元数据'}"
        )
        print("=" * 78)

    @staticmethod
    def _create_session(
        model_path: Path,
    ) -> "ort.InferenceSession":
        options = ort.SessionOptions()
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel
            .ORT_ENABLE_ALL
        )

        available = set(
            ort.get_available_providers()
        )

        plans: List[List[Any]] = []

        if "CUDAExecutionProvider" in available:
            plans.append([
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": "0",
                        "arena_extend_strategy":
                            "kNextPowerOfTwo",
                        "cudnn_conv_algo_search":
                            "EXHAUSTIVE",
                        "do_copy_in_default_stream":
                            "1",
                    },
                ),
                "CPUExecutionProvider",
            ])

        plans.append([
            "CPUExecutionProvider"
        ])

        errors = []

        for providers in plans:
            try:
                session = ort.InferenceSession(
                    str(model_path),
                    sess_options=options,
                    providers=providers,
                )

                actual = session.get_providers()

                requested_primary = (
                    providers[0][0]
                    if isinstance(
                        providers[0],
                        tuple,
                    )
                    else str(providers[0])
                )

                actual_primary = (
                    actual[0]
                    if actual
                    else "unknown"
                )

                if (
                    requested_primary
                    != "CPUExecutionProvider"
                    and actual_primary
                    == "CPUExecutionProvider"
                ):
                    errors.append(
                        f"{requested_primary} 请求后"
                        f"实际仍为 {actual}"
                    )
                    continue

                return session

            except Exception as exc:
                errors.append(
                    f"{providers}: {exc}"
                )

        raise RuntimeError(
            "normal.onnx Session 创建失败："
            + " | ".join(errors)
        )

    def _resolve_input_size(
        self,
    ) -> Tuple[int, int]:
        shape = list(self.input_info.shape)

        fallback = 640

        height = (
            int(shape[2])
            if (
                len(shape) >= 4
                and isinstance(
                    shape[2],
                    (int, np.integer),
                )
                and int(shape[2]) > 0
            )
            else fallback
        )

        width = (
            int(shape[3])
            if (
                len(shape) >= 4
                and isinstance(
                    shape[3],
                    (int, np.integer),
                )
                and int(shape[3]) > 0
            )
            else fallback
        )

        return height, width

    def _letterbox(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, LetterboxInfo]:
        source_h, source_w = image.shape[:2]

        ratio = min(
            self.input_width / source_w,
            self.input_height / source_h,
        )

        resized_w = max(
            1,
            int(round(source_w * ratio)),
        )
        resized_h = max(
            1,
            int(round(source_h * ratio)),
        )

        if (
            resized_w != source_w
            or resized_h != source_h
        ):
            resized = cv2.resize(
                image,
                (resized_w, resized_h),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            resized = image.copy()

        pad_x = (
            self.input_width - resized_w
        ) / 2.0

        pad_y = (
            self.input_height - resized_h
        ) / 2.0

        left = int(round(pad_x - 0.1))
        right = int(round(pad_x + 0.1))
        top = int(round(pad_y - 0.1))
        bottom = int(round(pad_y + 0.1))

        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        info = LetterboxInfo(
            ratio=ratio,
            pad_x=pad_x,
            pad_y=pad_y,
            input_width=self.input_width,
            input_height=self.input_height,
        )

        return padded, info

    def _preprocess(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, LetterboxInfo]:
        padded, info = self._letterbox(frame)

        rgb = cv2.cvtColor(
            padded,
            cv2.COLOR_BGR2RGB,
        )

        tensor = rgb.transpose(2, 0, 1)

        tensor = np.ascontiguousarray(
            tensor,
            dtype=np.float32,
        ) / 255.0

        return tensor[None], info

    @staticmethod
    def _nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        iou_threshold: float,
    ) -> List[int]:
        if len(boxes) == 0:
            return []

        keep_all: List[int] = []

        for class_id in np.unique(class_ids):
            class_indexes = np.where(
                class_ids == class_id
            )[0]

            class_boxes = boxes[class_indexes]
            class_scores = scores[class_indexes]

            x1 = class_boxes[:, 0]
            y1 = class_boxes[:, 1]
            x2 = class_boxes[:, 2]
            y2 = class_boxes[:, 3]

            areas = np.maximum(
                1.0,
                (x2 - x1) * (y2 - y1),
            )

            order = class_scores.argsort()[::-1]

            while order.size > 0:
                local_index = int(order[0])

                keep_all.append(
                    int(
                        class_indexes[
                            local_index
                        ]
                    )
                )

                if order.size == 1:
                    break

                xx1 = np.maximum(
                    x1[local_index],
                    x1[order[1:]],
                )
                yy1 = np.maximum(
                    y1[local_index],
                    y1[order[1:]],
                )
                xx2 = np.minimum(
                    x2[local_index],
                    x2[order[1:]],
                )
                yy2 = np.minimum(
                    y2[local_index],
                    y2[order[1:]],
                )

                intersection_w = np.maximum(
                    0.0,
                    xx2 - xx1,
                )
                intersection_h = np.maximum(
                    0.0,
                    yy2 - yy1,
                )

                intersection = (
                    intersection_w
                    * intersection_h
                )

                iou = intersection / np.maximum(
                    areas[local_index]
                    + areas[order[1:]]
                    - intersection,
                    1e-6,
                )

                remaining = np.where(
                    iou <= iou_threshold
                )[0]

                order = order[
                    remaining + 1
                ]

        keep_all.sort(
            key=lambda index: float(
                scores[index]
            ),
            reverse=True,
        )

        return keep_all

    def _restore_box(
        self,
        xyxy: Sequence[float],
        info: LetterboxInfo,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = [
            float(value)
            for value in xyxy
        ]

        x1 = (
            x1 - info.pad_x
        ) / info.ratio

        y1 = (
            y1 - info.pad_y
        ) / info.ratio

        x2 = (
            x2 - info.pad_x
        ) / info.ratio

        y2 = (
            y2 - info.pad_y
        ) / info.ratio

        x1 = max(
            0,
            min(frame_width - 1, int(round(x1))),
        )
        y1 = max(
            0,
            min(frame_height - 1, int(round(y1))),
        )
        x2 = max(
            0,
            min(frame_width - 1, int(round(x2))),
        )
        y2 = max(
            0,
            min(frame_height - 1, int(round(y2))),
        )

        return x1, y1, x2, y2

    @staticmethod
    def _looks_like_post_nms(
        prediction: np.ndarray,
    ) -> bool:
        if (
            prediction.ndim != 2
            or prediction.shape[1] != 6
            or len(prediction) == 0
        ):
            return False

        sample = prediction[
            :min(50, len(prediction))
        ]

        coords_valid = np.mean(
            (
                sample[:, 2] > sample[:, 0]
            )
            & (
                sample[:, 3] > sample[:, 1]
            )
        )

        scores_valid = np.mean(
            (
                sample[:, 4] >= 0.0
            )
            & (
                sample[:, 4] <= 1.01
            )
        )

        class_integer_like = np.mean(
            np.abs(
                sample[:, 5]
                - np.round(sample[:, 5])
            ) < 1e-3
        )

        return bool(
            coords_valid > 0.75
            and scores_valid > 0.90
            and class_integer_like > 0.90
        )

    def _parse_output(
        self,
        output: np.ndarray,
        info: LetterboxInfo,
        frame_width: int,
        frame_height: int,
    ) -> List[Detection]:
        prediction = np.asarray(output)

        prediction = np.squeeze(prediction)

        if prediction.ndim == 1:
            prediction = prediction.reshape(1, -1)

        # YOLOv8 常见 [84, 8400]，转成 [8400, 84]。
        if (
            prediction.ndim == 2
            and prediction.shape[0]
            < prediction.shape[1]
            and prediction.shape[0] <= 256
        ):
            prediction = prediction.T

        if (
            prediction.ndim != 2
            or prediction.shape[1] < 5
        ):
            return []

        boxes: List[List[float]] = []
        scores: List[float] = []
        class_ids: List[int] = []

        if self._looks_like_post_nms(
            prediction
        ):
            for row in prediction:
                score = float(row[4])

                if score < YOLO_CONFIDENCE:
                    continue

                boxes.append(
                    [
                        float(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                    ]
                )
                scores.append(score)
                class_ids.append(
                    int(round(float(row[5])))
                )

        else:
            class_count_from_metadata = (
                len(self.class_names)
            )

            column_count = prediction.shape[1]

            is_yolov5_style = bool(
                class_count_from_metadata > 0
                and column_count
                == 5 + class_count_from_metadata
            )

            for row in prediction:
                if row.shape[0] < 5:
                    continue

                xywh = row[:4].astype(
                    np.float32
                )

                if (
                    float(
                        np.max(
                            np.abs(xywh)
                        )
                    )
                    <= 2.0
                ):
                    xywh = xywh * np.asarray(
                        [
                            self.input_width,
                            self.input_height,
                            self.input_width,
                            self.input_height,
                        ],
                        dtype=np.float32,
                    )

                if row.shape[0] == 5:
                    score = float(row[4])
                    class_id = 0

                elif is_yolov5_style:
                    objectness = float(row[4])
                    class_scores = row[5:]

                    class_id = int(
                        np.argmax(class_scores)
                    )

                    score = (
                        objectness
                        * float(
                            class_scores[
                                class_id
                            ]
                        )
                    )

                else:
                    # YOLOv8: [x,y,w,h,cls...]
                    class_scores_v8 = row[4:]

                    class_id_v8 = int(
                        np.argmax(
                            class_scores_v8
                        )
                    )

                    score_v8 = float(
                        class_scores_v8[
                            class_id_v8
                        ]
                    )

                    # YOLOv5 兜底：
                    # [x,y,w,h,obj,cls...]
                    score_v5 = -1.0
                    class_id_v5 = 0

                    if row.shape[0] >= 6:
                        class_scores_v5 = row[5:]

                        class_id_v5 = int(
                            np.argmax(
                                class_scores_v5
                            )
                        )

                        score_v5 = (
                            float(row[4])
                            * float(
                                class_scores_v5[
                                    class_id_v5
                                ]
                            )
                        )

                    if score_v5 > score_v8:
                        score = score_v5
                        class_id = class_id_v5
                    else:
                        score = score_v8
                        class_id = class_id_v8

                if score < YOLO_CONFIDENCE:
                    continue

                center_x, center_y, box_w, box_h = [
                    float(value)
                    for value in xywh
                ]

                boxes.append([
                    center_x - box_w / 2.0,
                    center_y - box_h / 2.0,
                    center_x + box_w / 2.0,
                    center_y + box_h / 2.0,
                ])

                scores.append(score)
                class_ids.append(class_id)

        if not boxes:
            return []

        boxes_array = np.asarray(
            boxes,
            dtype=np.float32,
        )

        scores_array = np.asarray(
            scores,
            dtype=np.float32,
        )

        class_array = np.asarray(
            class_ids,
            dtype=np.int32,
        )

        keep = self._nms(
            boxes_array,
            scores_array,
            class_array,
            YOLO_NMS_IOU,
        )

        detections: List[Detection] = []

        for index in keep:
            restored_box = self._restore_box(
                boxes_array[index],
                info,
                frame_width,
                frame_height,
            )

            x1, y1, x2, y2 = restored_box

            if (
                x2 - x1 < MIN_BOX_WIDTH
                or y2 - y1 < MIN_BOX_HEIGHT
            ):
                continue

            class_id = int(
                class_array[index]
            )

            class_name = str(
                self.class_names.get(
                    class_id,
                    f"class_{class_id}",
                )
            )

            detections.append(
                Detection(
                    bbox=restored_box,
                    confidence=float(
                        scores_array[index]
                    ),
                    class_id=class_id,
                    class_name=class_name,
                )
            )

        return detections

    def detect(
        self,
        frame: np.ndarray,
    ) -> Tuple[List[Detection], float]:
        tensor, info = self._preprocess(frame)

        start = time.perf_counter()

        outputs = self.session.run(
            self.output_names or None,
            {
                self.input_name: tensor,
            },
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        if not outputs:
            return [], elapsed_ms

        detections = self._parse_output(
            outputs[0],
            info,
            frame.shape[1],
            frame.shape[0],
        )

        return detections, elapsed_ms

    def is_vehicle_detection(
        self,
        detection: Detection,
    ) -> bool:
        if TREAT_ALL_DETECTIONS_AS_VEHICLES:
            return True

        if FORCE_VEHICLE_CLASS_IDS is not None:
            return (
                detection.class_id
                in FORCE_VEHICLE_CLASS_IDS
            )

        normalized = normalize_class_name(
            detection.class_name
        )

        if normalized in VEHICLE_NAME_KEYS:
            return True

        # 模型只有一个类别并且没有类别元数据时，
        # 按“车辆单类别模型”处理。
        if (
            not self.class_names
            and detection.class_id == 0
        ):
            return True

        return False


# ============================================================
# 车辆 Mask
# ============================================================

def box_road_intersection_ratio(
    bbox: Tuple[int, int, int, int],
    road_mask: np.ndarray,
) -> float:
    x1, y1, x2, y2 = bbox

    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    box_area = box_width * box_height

    crop = road_mask[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        return 0.0

    road_pixels = int(
        np.count_nonzero(crop)
    )

    return road_pixels / float(
        max(box_area, 1)
    )


def bottom_center_inside_road(
    bbox: Tuple[int, int, int, int],
    road_mask: np.ndarray,
) -> bool:
    x1, y1, x2, y2 = bbox

    center_x = int(round((x1 + x2) / 2.0))
    bottom_y = int(round(y2 - 1))

    center_x = max(
        0,
        min(
            road_mask.shape[1] - 1,
            center_x,
        ),
    )

    bottom_y = max(
        0,
        min(
            road_mask.shape[0] - 1,
            bottom_y,
        ),
    )

    return bool(
        road_mask[bottom_y, center_x] > 0
    )


def expand_vehicle_box(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    expand_x = int(
        round(box_w * VEHICLE_MASK_EXPAND_X)
    )

    expand_top = int(
        round(box_h * VEHICLE_MASK_EXPAND_TOP)
    )

    expand_bottom = int(
        round(box_h * VEHICLE_MASK_EXPAND_BOTTOM)
    )

    return (
        max(0, x1 - expand_x),
        max(0, y1 - expand_top),
        min(width - 1, x2 + expand_x),
        min(height - 1, y2 + expand_bottom),
    )


def build_vehicle_mask(
    detections: List[Detection],
    detector: NormalOnnxVehicleDetector,
    road_mask: np.ndarray,
) -> Tuple[
    np.ndarray,
    List[Tuple[Detection, Tuple[int, int, int, int], float]],
]:
    height, width = road_mask.shape[:2]

    vehicle_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    accepted = []

    for detection in detections:
        if not detector.is_vehicle_detection(
            detection
        ):
            continue

        intersection_ratio = (
            box_road_intersection_ratio(
                detection.bbox,
                road_mask,
            )
        )

        bottom_inside = (
            bottom_center_inside_road(
                detection.bbox,
                road_mask,
            )
        )

        if (
            intersection_ratio
            < MIN_BOX_ROAD_INTERSECTION
            and not bottom_inside
        ):
            continue

        expanded_box = expand_vehicle_box(
            detection.bbox,
            width,
            height,
        )

        x1, y1, x2, y2 = expanded_box

        cv2.rectangle(
            vehicle_mask,
            (x1, y1),
            (x2, y2),
            255,
            -1,
        )

        accepted.append(
            (
                detection,
                expanded_box,
                intersection_ratio,
            )
        )

    # 车辆忽略区域最终必须限制在道路内部。
    vehicle_mask = cv2.bitwise_and(
        vehicle_mask,
        road_mask,
    )

    return vehicle_mask, accepted


# ============================================================
# 可视化
# ============================================================

def create_mask_inset(
    road_mask: np.ndarray,
    vehicle_mask: np.ndarray,
    valid_mask: np.ndarray,
    max_width: int,
) -> np.ndarray:
    height, width = road_mask.shape[:2]

    panel = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    # 道路区域：深灰
    panel[road_mask > 0] = (
        70,
        70,
        70,
    )

    # 有效检测道路：绿色
    panel[valid_mask > 0] = (
        50,
        190,
        80,
    )

    # 车辆忽略区域：红色
    panel[vehicle_mask > 0] = (
        50,
        50,
        230,
    )

    scale = min(
        1.0,
        max_width / max(width, 1),
    )

    resized = cv2.resize(
        panel,
        (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    cv2.rectangle(
        resized,
        (0, 0),
        (resized.shape[1] - 1, resized.shape[0] - 1),
        (245, 245, 245),
        2,
    )

    cv2.putText(
        resized,
        "Effective road mask",
        (9, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return resized


def draw_result(
    frame: np.ndarray,
    road_polygon: np.ndarray,
    road_mask: np.ndarray,
    vehicle_mask: np.ndarray,
    valid_mask: np.ndarray,
    accepted_detections: List[
        Tuple[
            Detection,
            Tuple[int, int, int, int],
            float,
        ]
    ],
    roi_diagnostics: Dict[str, Any],
    roi_fallback: bool,
    provider_name: str,
    inference_ms: float,
    processing_fps: float,
    frame_index: int,
    total_frames: int,
    paused: bool,
) -> np.ndarray:
    output = frame.copy()

    # 先轻微显示道路 ROI。
    road_layer = output.copy()
    road_layer[road_mask > 0] = (
        50,
        170,
        110,
    )

    output = cv2.addWeighted(
        road_layer,
        0.13,
        output,
        0.87,
        0,
    )

    # 车辆忽略区域用红色透明覆盖。
    vehicle_layer = output.copy()
    vehicle_layer[vehicle_mask > 0] = (
        35,
        35,
        235,
    )

    output = cv2.addWeighted(
        vehicle_layer,
        0.30,
        output,
        0.70,
        0,
    )

    polygon_int = np.round(
        road_polygon
    ).astype(np.int32).reshape((-1, 1, 2))

    cv2.polylines(
        output,
        [polygon_int],
        True,
        (0, 225, 255),
        4,
        cv2.LINE_AA,
    )

    # 车辆原始框 + 扩展 Mask 框。
    for (
        detection,
        expanded_box,
        intersection_ratio,
    ) in accepted_detections:
        x1, y1, x2, y2 = detection.bbox
        ex1, ey1, ex2, ey2 = expanded_box

        cv2.rectangle(
            output,
            (ex1, ey1),
            (ex2, ey2),
            (40, 40, 220),
            2,
            cv2.LINE_AA,
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (40, 220, 255),
            3,
            cv2.LINE_AA,
        )

        label = (
            f"{detection.class_name} "
            f"{detection.confidence:.2f} "
            f"road={intersection_ratio:.2f}"
        )

        text_y = max(22, y1 - 8)

        cv2.putText(
            output,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    # 右上角 Mask 小图。
    inset = create_mask_inset(
        road_mask,
        vehicle_mask,
        valid_mask,
        max_width=max(
            260,
            int(round(output.shape[1] * 0.28)),
        ),
    )

    inset_x = output.shape[1] - inset.shape[1] - 15
    inset_y = 112

    if (
        inset_x >= 0
        and inset_y + inset.shape[0]
        <= output.shape[0]
    ):
        output[
            inset_y:inset_y + inset.shape[0],
            inset_x:inset_x + inset.shape[1],
        ] = inset

    # 顶部状态栏。
    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 100),
        (9, 16, 25),
        -1,
    )

    state = "PAUSED" if paused else "PLAYING"

    fallback_text = (
        "fallback"
        if roi_fallback
        else "tracked"
    )

    line1 = (
        f"{state} | Frame "
        f"{frame_index}/"
        f"{total_frames if total_frames > 0 else '?'} "
        f"| FPS {processing_fps:.1f}"
    )

    line2 = (
        f"normal.onnx: {provider_name} "
        f"| infer {inference_ms:.1f} ms "
        f"| road vehicles "
        f"{len(accepted_detections)}"
    )

    line3 = (
        f"ROI: {fallback_text} "
        f"| matches "
        f"{roi_diagnostics.get('good_matches', 0)} "
        f"| inliers "
        f"{roi_diagnostics.get('inliers', 0)} "
        f"| valid road "
        f"{100.0 * np.count_nonzero(valid_mask) / max(valid_mask.size, 1):.1f}%"
    )

    for text, y, color in (
        (
            line1,
            27,
            (245, 245, 245),
        ),
        (
            line2,
            58,
            (80, 230, 255),
        ),
        (
            line3,
            88,
            (125, 240, 150),
        ),
    ):
        cv2.putText(
            output,
            text,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def resize_to_fit(
    image: np.ndarray,
    max_width: int,
    max_height: int,
) -> np.ndarray:
    height, width = image.shape[:2]

    scale = min(
        max_width / max(width, 1),
        max_height / max(height, 1),
        1.0,
    )

    if scale >= 1.0:
        return image

    return cv2.resize(
        image,
        (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    project_root = Path(__file__).resolve().parent

    video_path = project_root / VIDEO_FILENAME
    model_path = project_root / MODEL_FILENAME
    anchor_image_path = (
        project_root / ANCHOR_IMAGE_PATH
    )
    roi_json_path = (
        project_root / ROAD_ROI_JSON_PATH
    )
    anchor_mask_path = (
        project_root / ANCHOR_ROAD_MASK_PATH
    )
    output_video_path = (
        project_root / OUTPUT_VIDEO_PATH
    )

    for required_path, label in (
        (video_path, "正常道路视频"),
        (model_path, "normal.onnx"),
        (anchor_image_path, "reference_01.png"),
        (roi_json_path, "road_roi.json"),
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"没有找到{label}：{required_path}"
            )

    anchor_image = imread_unicode(
        anchor_image_path,
        cv2.IMREAD_COLOR,
    )

    if anchor_image is None:
        raise RuntimeError(
            f"无法读取基准图：{anchor_image_path}"
        )

    anchor_polygon = load_roi_points(
        roi_json_path
    )

    anchor_road_mask = None

    if anchor_mask_path.exists():
        anchor_road_mask = imread_unicode(
            anchor_mask_path,
            cv2.IMREAD_GRAYSCALE,
        )

    if (
        anchor_road_mask is None
        or anchor_road_mask.shape[:2]
        != anchor_image.shape[:2]
    ):
        anchor_road_mask = polygon_to_mask(
            anchor_polygon,
            anchor_image.shape[1],
            anchor_image.shape[0],
        )

    detector = NormalOnnxVehicleDetector(
        model_path
    )

    roi_tracker = RoadRoiTracker(
        anchor_image=anchor_image,
        anchor_polygon=anchor_polygon,
        anchor_road_mask=anchor_road_mask,
    )

    capture, temp_video_path = (
        open_video_with_fallback(video_path)
    )

    writer = None

    try:
        source_fps = float(
            capture.get(cv2.CAP_PROP_FPS)
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

        anchor_height, anchor_width = (
            anchor_image.shape[:2]
        )

        if SAVE_OUTPUT_VIDEO:
            output_video_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            writer = cv2.VideoWriter(
                str(output_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                source_fps,
                (anchor_width, anchor_height),
            )

            if not writer.isOpened():
                raise RuntimeError(
                    f"无法创建输出视频：{output_video_path}"
                )

        print("=" * 78)
        print("道路 ROI + normal.onnx 车辆忽略 Mask")
        print(f"视频：{video_path}")
        print(f"模型：{model_path}")
        print(f"道路标定：{roi_json_path}")
        print(f"输出：{output_video_path}")
        print("=" * 78)

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        paused = False
        single_step = False
        frame_index = 0

        last_frame: Optional[np.ndarray] = None
        last_road_polygon = anchor_polygon.copy()
        last_roi_fallback = False
        last_roi_diagnostics: Dict[str, Any] = {
            "status": "anchor",
            "good_matches": 0,
            "inliers": 0,
        }

        last_detections: List[Detection] = []
        last_inference_ms = 0.0

        processing_fps_ema = 0.0

        while True:
            should_read = (
                not paused
                or single_step
                or last_frame is None
            )

            new_frame_read = False

            if should_read:
                success, source_frame = capture.read()

                if (
                    not success
                    or source_frame is None
                ):
                    print("视频播放结束。")
                    break

                frame_index += 1
                new_frame_read = True

                frame = cv2.resize(
                    source_frame,
                    (anchor_width, anchor_height),
                    interpolation=cv2.INTER_AREA,
                )

                last_frame = frame
                loop_start = time.perf_counter()

                # 1. 先更新道路 ROI。
                if (
                    frame_index == 1
                    or frame_index
                    % ROI_UPDATE_EVERY_N_FRAMES
                    == 0
                ):
                    (
                        last_road_polygon,
                        last_roi_fallback,
                        last_roi_diagnostics,
                    ) = roi_tracker.update(frame)

                road_mask = polygon_to_mask(
                    last_road_polygon,
                    anchor_width,
                    anchor_height,
                )

                # 2. 再运行 normal.onnx 检测车辆。
                if (
                    frame_index == 1
                    or frame_index
                    % YOLO_EVERY_N_FRAMES
                    == 0
                ):
                    (
                        last_detections,
                        last_inference_ms,
                    ) = detector.detect(frame)

                # 3. 道路内车辆转成忽略 Mask。
                (
                    vehicle_mask,
                    accepted_detections,
                ) = build_vehicle_mask(
                    last_detections,
                    detector,
                    road_mask,
                )

                # 4. 最终有效区域。
                valid_mask = cv2.bitwise_and(
                    road_mask,
                    cv2.bitwise_not(
                        vehicle_mask
                    ),
                )

                elapsed = (
                    time.perf_counter()
                    - loop_start
                )

                instant_fps = (
                    1.0 / elapsed
                    if elapsed > 0
                    else 0.0
                )

                if processing_fps_ema <= 0:
                    processing_fps_ema = instant_fps
                else:
                    processing_fps_ema = (
                        0.90 * processing_fps_ema
                        + 0.10 * instant_fps
                    )

                result = draw_result(
                    frame=frame,
                    road_polygon=last_road_polygon,
                    road_mask=road_mask,
                    vehicle_mask=vehicle_mask,
                    valid_mask=valid_mask,
                    accepted_detections=accepted_detections,
                    roi_diagnostics=last_roi_diagnostics,
                    roi_fallback=last_roi_fallback,
                    provider_name=(
                        detector.providers[0]
                        if detector.providers
                        else "unknown"
                    ),
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
                # 暂停时沿用上一次结果，不继续写输出视频。
                if last_frame is None:
                    continue

                road_mask = polygon_to_mask(
                    last_road_polygon,
                    anchor_width,
                    anchor_height,
                )

                (
                    vehicle_mask,
                    accepted_detections,
                ) = build_vehicle_mask(
                    last_detections,
                    detector,
                    road_mask,
                )

                valid_mask = cv2.bitwise_and(
                    road_mask,
                    cv2.bitwise_not(
                        vehicle_mask
                    ),
                )

                result = draw_result(
                    frame=last_frame,
                    road_polygon=last_road_polygon,
                    road_mask=road_mask,
                    vehicle_mask=vehicle_mask,
                    valid_mask=valid_mask,
                    accepted_detections=accepted_detections,
                    roi_diagnostics=last_roi_diagnostics,
                    roi_fallback=last_roi_fallback,
                    provider_name=(
                        detector.providers[0]
                        if detector.providers
                        else "unknown"
                    ),
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

            key = cv2.waitKey(delay)

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

    print("=" * 78)
    print("处理结束")
    print(f"可视化视频：{output_video_path}")
    print("=" * 78)
    print(
        "绿色：道路 ROI；红色：车辆忽略区域；"
        "右上角绿色区域：后续真正参与障碍物比较的道路。"
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
            f"[失败] {type(exc).__name__}: {exc}"
        )
        raise

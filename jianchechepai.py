# -*- coding: utf-8 -*-
"""
第七步：本地视频实时连续车道线检测

流程：
1. 运行后弹出窗口选择本地视频，不需要修改代码路径。
2. 读取视频第一帧，用鼠标圈出道路 ROI。
3. 每帧执行：
   灰度化 -> 高斯模糊 -> Canny -> ROI -> HoughLinesP
   -> 近竖直线过滤 -> 空间连续性聚类 -> 车道轨迹拟合
   -> 跨帧匹配与指数平滑
4. 实时显示结果，不保存任何文件。

空间连续性用于过滤：
- 方向箭头
- 车辆轮廓
- 短小噪声线
- 只在局部出现的伪直线

时间连续性用于：
- 减少车道线闪烁
- 平滑线条抖动
- 允许短时间被车辆遮挡

ROI 操作：
- 鼠标左键：添加顶点
- 鼠标右键 / Backspace：撤销
- R：清空
- Enter：确认
- Esc：取消

播放窗口：
- Q / Esc：退出
- Space：暂停 / 继续
- R：在当前帧重新选择 ROI
- S：单步前进一帧（暂停状态下）
- 方向键右：快进约 5 秒
- 方向键左：后退约 5 秒

程序不会保存视频或图片。
"""

import math
import sys
import time
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# 窗口与显示参数
# ============================================================

ROI_WINDOW = "Select Road ROI"
VIDEO_WINDOW = "Real-time Continuous Lane Detection"

MAX_ROI_SELECT_WIDTH = 1500
MAX_ROI_SELECT_HEIGHT = 850

MAX_DISPLAY_WIDTH = 1800
MAX_DISPLAY_HEIGHT = 940

# 为提高实时性，较宽视频会按比例缩小后处理。
# ROI 也是在处理分辨率下选择和保存的。
PROCESS_MAX_WIDTH = 1280

# 1 表示每帧处理；2 表示隔一帧处理一次。
PROCESS_EVERY_N_FRAMES = 1


# ============================================================
# 图像预处理参数
# ============================================================

GAUSSIAN_KERNEL = (5, 5)
CANNY_LOW = 50
CANNY_HIGH = 150


# ============================================================
# HoughLinesP 参数
# ============================================================

HOUGH_THRESHOLD = 32
HOUGH_MIN_LINE_LENGTH = 24
HOUGH_MAX_LINE_GAP = 16

# 为避免候选过多导致聚类变慢，只保留最长的一部分。
MAX_INITIAL_SEGMENTS = 140


# ============================================================
# 近竖直线段初筛
# ============================================================

INITIAL_MIN_LENGTH = 30.0

# 角度相对水平线：
# 0° 水平，90° 竖直。
INITIAL_MIN_ANGLE = 56.0
INITIAL_MAX_ANGLE = 90.0


# ============================================================
# 单帧空间连续性参数
# ============================================================

MAX_ANGLE_DIFFERENCE = 9.0
MAX_SLOPE_DIFFERENCE = 0.22

MIN_LATERAL_DISTANCE = 16.0
LATERAL_DISTANCE_RATIO = 0.018

MIN_VERTICAL_GAP = 55.0
VERTICAL_GAP_RATIO = 0.16

MIN_SUPPORT_SEGMENTS = 3
MIN_VERTICAL_SPAN_RATIO = 0.34

VERTICAL_BAND_COUNT = 8
MIN_OCCUPIED_BANDS = 4

MAX_START_POSITION_RATIO = 0.45
MIN_END_POSITION_RATIO = 0.62
MIN_RAW_COVERAGE_RATIO = 0.23

MIN_FIT_RESIDUAL = 11.0
FIT_RESIDUAL_WIDTH_RATIO = 0.012

# 一条非常长的实线即使只有一个 Hough 片段，也允许通过。
SINGLE_LONG_LINE_SPAN_RATIO = 0.62

# 同一条白线的左右边缘去重。
MIN_DUPLICATE_DISTANCE = 14.0
DUPLICATE_DISTANCE_RATIO = 0.014


# ============================================================
# 跨帧时间稳定参数
# ============================================================

# 新检测结果与旧轨迹在 ROI 下端的横向距离小于此值时可匹配。
TEMPORAL_MATCH_MIN_DISTANCE = 34.0
TEMPORAL_MATCH_WIDTH_RATIO = 0.035
TEMPORAL_MAX_SLOPE_DIFFERENCE = 0.30

# 指数平滑系数。
# 越小越稳定但响应越慢；越大越灵敏但更容易抖动。
TEMPORAL_EMA_ALPHA = 0.32

# 连续命中多少次后视为正式车道线。
TEMPORAL_MIN_CONFIRM_HITS = 2

# 最多允许连续丢失多少帧，适合车辆短暂遮挡。
TEMPORAL_MAX_MISSED_FRAMES = 6


# ============================================================
# 数据结构
# ============================================================

@dataclass
class LaneSegment:
    x1: int
    y1: int
    x2: int
    y2: int
    length: float
    angle: float
    slope_xy: float
    intercept_x: float
    y_min: float
    y_max: float

    def x_at(self, y: float) -> float:
        return self.slope_xy * y + self.intercept_x

    @property
    def y_mid(self) -> float:
        return (self.y_min + self.y_max) / 2.0


@dataclass
class LaneTrack:
    segments: List[LaneSegment]
    slope_xy: float
    intercept_x: float
    y_min: float
    y_max: float
    vertical_span: float
    coverage_ratio: float
    occupied_bands: int
    support_count: int
    residual_mean: float
    score: float

    def x_at(self, y: float) -> float:
        return self.slope_xy * y + self.intercept_x


@dataclass
class TemporalTrack:
    track_id: int
    slope_xy: float
    intercept_x: float
    y_min: float
    y_max: float
    score: float
    support_count: int
    occupied_bands: int
    hits: int = 1
    missed: int = 0
    age: int = 1
    matched_this_frame: bool = True

    def x_at(self, y: float) -> float:
        return self.slope_xy * y + self.intercept_x


# ============================================================
# 文件选择与图像辅助
# ============================================================

def choose_video() -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.askopenfilename(
        title="请选择需要检测的本地道路视频",
        filetypes=[
            ("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.m4v"),
            ("MP4 视频", "*.mp4"),
            ("所有文件", "*.*"),
        ],
    )

    root.destroy()
    return path


def resize_to_max_width(image, max_width: int):
    h, w = image.shape[:2]

    if w <= max_width:
        return image.copy(), 1.0

    scale = max_width / float(w)
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(
        image,
        (max_width, new_h),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def resize_to_fit(image, max_width: int, max_height: int):
    h, w = image.shape[:2]

    scale = min(
        max_width / max(w, 1),
        max_height / max(h, 1),
        1.0,
    )

    if scale >= 1.0:
        return image.copy(), 1.0

    resized = cv2.resize(
        image,
        (
            max(1, int(round(w * scale))),
            max(1, int(round(h * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def add_title(image, title: str):
    if image.ndim == 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR,
        )
    else:
        image = image.copy()

    canvas = cv2.copyMakeBorder(
        image,
        44,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(18, 20, 24),
    )

    cv2.putText(
        canvas,
        title,
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    return canvas


# ============================================================
# 手动 ROI 选择
# ============================================================

class ROISelector:
    def __init__(self, frame):
        self.original = frame
        self.preview, self.display_scale = resize_to_fit(
            frame,
            MAX_ROI_SELECT_WIDTH,
            MAX_ROI_SELECT_HEIGHT,
        )
        self.points: List[Tuple[int, int]] = []
        self.confirmed = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((int(x), int(y)))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()

    def render(self):
        canvas = cv2.addWeighted(
            self.preview,
            0.82,
            np.zeros_like(self.preview),
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
                    (40, 170, 220),
                )

                canvas = cv2.addWeighted(
                    layer,
                    0.24,
                    canvas,
                    0.76,
                    0,
                )

                cv2.polylines(
                    canvas,
                    [pts],
                    True,
                    (0, 230, 255),
                    3,
                    cv2.LINE_AA,
                )

            elif len(self.points) >= 2:
                cv2.polylines(
                    canvas,
                    [pts],
                    False,
                    (0, 230, 255),
                    3,
                    cv2.LINE_AA,
                )

            for index, point in enumerate(
                self.points,
                start=1,
            ):
                cv2.circle(
                    canvas,
                    point,
                    7,
                    (0, 255, 255),
                    -1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    canvas,
                    str(index),
                    (point[0] + 9, point[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        cv2.rectangle(
            canvas,
            (0, 0),
            (canvas.shape[1], 76),
            (15, 20, 28),
            -1,
        )

        cv2.putText(
            canvas,
            "Left click:add | Right click/Backspace:undo | R:reset",
            (18, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (240, 245, 250),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"Enter:confirm | ESC:cancel | Points:{len(self.points)}",
            (18, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (70, 220, 255),
            2,
            cv2.LINE_AA,
        )

        return canvas

    def select(self) -> Optional[np.ndarray]:
        cv2.namedWindow(
            ROI_WINDOW,
            cv2.WINDOW_NORMAL,
        )

        cv2.setMouseCallback(
            ROI_WINDOW,
            self.mouse_callback,
        )

        while True:
            cv2.imshow(
                ROI_WINDOW,
                self.render(),
            )

            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10):
                if len(self.points) >= 3:
                    self.confirmed = True
                    break

                print("至少需要选择 3 个 ROI 顶点。")

            elif key in (8, 127):
                if self.points:
                    self.points.pop()

            elif key in (ord("r"), ord("R")):
                self.points.clear()

            elif key == 27:
                break

            try:
                if cv2.getWindowProperty(
                    ROI_WINDOW,
                    cv2.WND_PROP_VISIBLE,
                ) < 1:
                    break
            except cv2.error:
                break

        cv2.destroyWindow(ROI_WINDOW)

        if not self.confirmed:
            return None

        h, w = self.original.shape[:2]
        converted = []

        for x, y in self.points:
            original_x = int(
                round(x / self.display_scale)
            )
            original_y = int(
                round(y / self.display_scale)
            )

            original_x = max(
                0,
                min(w - 1, original_x),
            )
            original_y = max(
                0,
                min(h - 1, original_y),
            )

            converted.append([
                original_x,
                original_y,
            ])

        return np.array(
            converted,
            dtype=np.int32,
        )


# ============================================================
# Hough 线段初筛
# ============================================================

def create_segment(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Optional[LaneSegment]:
    if y1 > y2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    dx = float(x2 - x1)
    dy = float(y2 - y1)

    if abs(dy) < 1.0:
        return None

    length = math.hypot(dx, dy)

    angle = math.degrees(
        math.atan2(
            abs(dy),
            abs(dx) + 1e-9,
        )
    )

    if length < INITIAL_MIN_LENGTH:
        return None

    if not (
        INITIAL_MIN_ANGLE
        <= angle
        <= INITIAL_MAX_ANGLE
    ):
        return None

    slope_xy = dx / dy
    intercept_x = float(x1) - slope_xy * float(y1)

    return LaneSegment(
        x1=int(x1),
        y1=int(y1),
        x2=int(x2),
        y2=int(y2),
        length=float(length),
        angle=float(angle),
        slope_xy=float(slope_xy),
        intercept_x=float(intercept_x),
        y_min=float(y1),
        y_max=float(y2),
    )


def extract_initial_segments(raw_lines) -> List[LaneSegment]:
    segments: List[LaneSegment] = []

    if raw_lines is None:
        return segments

    for raw in raw_lines:
        x1, y1, x2, y2 = map(
            int,
            raw[0],
        )

        segment = create_segment(
            x1,
            y1,
            x2,
            y2,
        )

        if segment is not None:
            segments.append(segment)

    segments.sort(
        key=lambda item: item.length,
        reverse=True,
    )

    return segments[:MAX_INITIAL_SEGMENTS]


# ============================================================
# 空间连续性聚类
# ============================================================

class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[
                self.parent[index]
            ]
            index = self.parent[index]

        return index

    def union(self, first: int, second: int):
        root_first = self.find(first)
        root_second = self.find(second)

        if root_first == root_second:
            return

        if self.rank[root_first] < self.rank[root_second]:
            self.parent[root_first] = root_second

        elif self.rank[root_first] > self.rank[root_second]:
            self.parent[root_second] = root_first

        else:
            self.parent[root_second] = root_first
            self.rank[root_first] += 1


def vertical_gap(
    first: LaneSegment,
    second: LaneSegment,
) -> float:
    if first.y_max < second.y_min:
        return second.y_min - first.y_max

    if second.y_max < first.y_min:
        return first.y_min - second.y_max

    return 0.0


def lateral_distance(
    first: LaneSegment,
    second: LaneSegment,
) -> float:
    overlap_start = max(
        first.y_min,
        second.y_min,
    )
    overlap_end = min(
        first.y_max,
        second.y_max,
    )

    if overlap_start <= overlap_end:
        probe_y = (
            overlap_start + overlap_end
        ) / 2.0

    elif first.y_max < second.y_min:
        probe_y = (
            first.y_max + second.y_min
        ) / 2.0

    else:
        probe_y = (
            second.y_max + first.y_min
        ) / 2.0

    return abs(
        first.x_at(probe_y)
        - second.x_at(probe_y)
    )


def segments_are_continuous(
    first: LaneSegment,
    second: LaneSegment,
    max_lateral_distance: float,
    max_vertical_gap: float,
) -> bool:
    if abs(
        first.angle - second.angle
    ) > MAX_ANGLE_DIFFERENCE:
        return False

    if abs(
        first.slope_xy - second.slope_xy
    ) > MAX_SLOPE_DIFFERENCE:
        return False

    gap = vertical_gap(
        first,
        second,
    )

    if gap > max_vertical_gap:
        return False

    lateral = lateral_distance(
        first,
        second,
    )

    gap_bonus = min(
        max_lateral_distance * 0.35,
        gap * 0.08,
    )

    return lateral <= (
        max_lateral_distance + gap_bonus
    )


def cluster_segments(
    segments: List[LaneSegment],
    image_width: int,
    roi_height: float,
) -> List[List[LaneSegment]]:
    if not segments:
        return []

    max_lateral_distance = max(
        MIN_LATERAL_DISTANCE,
        image_width * LATERAL_DISTANCE_RATIO,
    )

    max_vertical_gap = max(
        MIN_VERTICAL_GAP,
        roi_height * VERTICAL_GAP_RATIO,
    )

    union_find = UnionFind(
        len(segments)
    )

    for first_index in range(len(segments)):
        for second_index in range(
            first_index + 1,
            len(segments),
        ):
            if segments_are_continuous(
                segments[first_index],
                segments[second_index],
                max_lateral_distance,
                max_vertical_gap,
            ):
                union_find.union(
                    first_index,
                    second_index,
                )

    grouped: Dict[int, List[LaneSegment]] = {}

    for index, segment in enumerate(segments):
        root = union_find.find(index)
        grouped.setdefault(
            root,
            [],
        ).append(segment)

    return list(grouped.values())


# ============================================================
# 轨迹拟合与验证
# ============================================================

def weighted_fit_x_from_y(
    segments: List[LaneSegment],
) -> Tuple[float, float]:
    y_values = []
    x_values = []
    weights = []

    for segment in segments:
        weight = math.sqrt(
            max(segment.length, 1.0)
        )

        y_values.extend([
            float(segment.y1),
            float(segment.y2),
        ])

        x_values.extend([
            float(segment.x1),
            float(segment.x2),
        ])

        weights.extend([
            weight,
            weight,
        ])

    coefficients = np.polyfit(
        np.asarray(
            y_values,
            dtype=np.float64,
        ),
        np.asarray(
            x_values,
            dtype=np.float64,
        ),
        deg=1,
        w=np.asarray(
            weights,
            dtype=np.float64,
        ),
    )

    return (
        float(coefficients[0]),
        float(coefficients[1]),
    )


def segment_fit_residual(
    segment: LaneSegment,
    slope_xy: float,
    intercept_x: float,
) -> float:
    probe_y_values = [
        float(segment.y1),
        (segment.y_min + segment.y_max) / 2.0,
        float(segment.y2),
    ]

    residuals = []

    for probe_y in probe_y_values:
        predicted_x = (
            slope_xy * probe_y
            + intercept_x
        )

        actual_x = segment.x_at(
            probe_y
        )

        residuals.append(
            abs(actual_x - predicted_x)
        )

    return float(
        np.mean(residuals)
    )


def robust_fit_cluster(
    segments: List[LaneSegment],
    image_width: int,
):
    working = list(segments)

    residual_limit = max(
        MIN_FIT_RESIDUAL,
        image_width * FIT_RESIDUAL_WIDTH_RATIO,
    )

    for _ in range(3):
        if not working:
            break

        slope_xy, intercept_x = weighted_fit_x_from_y(
            working
        )

        filtered = [
            segment
            for segment in working
            if segment_fit_residual(
                segment,
                slope_xy,
                intercept_x,
            ) <= residual_limit
        ]

        if len(filtered) == len(working):
            working = filtered
            break

        if not filtered:
            break

        working = filtered

    if not working:
        return [], 0.0, 0.0, float("inf")

    slope_xy, intercept_x = weighted_fit_x_from_y(
        working
    )

    residual_mean = float(
        np.mean([
            segment_fit_residual(
                segment,
                slope_xy,
                intercept_x,
            )
            for segment in working
        ])
    )

    return (
        working,
        slope_xy,
        intercept_x,
        residual_mean,
    )


def merge_intervals(
    intervals: List[Tuple[float, float]],
):
    if not intervals:
        return []

    sorted_intervals = sorted(
        intervals,
        key=lambda item: item[0],
    )

    merged = [
        [
            float(sorted_intervals[0][0]),
            float(sorted_intervals[0][1]),
        ]
    ]

    for start, end in sorted_intervals[1:]:
        last = merged[-1]

        if start <= last[1]:
            last[1] = max(
                last[1],
                float(end),
            )
        else:
            merged.append([
                float(start),
                float(end),
            ])

    return [
        (start, end)
        for start, end in merged
    ]


def calculate_occupied_bands(
    segments: List[LaneSegment],
    roi_y_min: float,
    roi_height: float,
) -> int:
    if roi_height <= 0:
        return 0

    band_height = (
        roi_height
        / VERTICAL_BAND_COUNT
    )

    occupied = set()

    for segment in segments:
        start_index = int(
            math.floor(
                (
                    segment.y_min
                    - roi_y_min
                )
                / band_height
            )
        )

        end_index = int(
            math.floor(
                (
                    segment.y_max
                    - roi_y_min
                )
                / band_height
            )
        )

        start_index = max(
            0,
            min(
                VERTICAL_BAND_COUNT - 1,
                start_index,
            ),
        )

        end_index = max(
            0,
            min(
                VERTICAL_BAND_COUNT - 1,
                end_index,
            ),
        )

        for band_index in range(
            start_index,
            end_index + 1,
        ):
            occupied.add(band_index)

    return len(occupied)


def build_lane_track(
    raw_cluster: List[LaneSegment],
    image_width: int,
    roi_y_min: float,
    roi_height: float,
) -> Optional[LaneTrack]:
    (
        fitted_segments,
        slope_xy,
        intercept_x,
        residual_mean,
    ) = robust_fit_cluster(
        raw_cluster,
        image_width,
    )

    if not fitted_segments:
        return None

    y_min = min(
        segment.y_min
        for segment in fitted_segments
    )

    y_max = max(
        segment.y_max
        for segment in fitted_segments
    )

    vertical_span = max(
        0.0,
        y_max - y_min,
    )

    intervals = merge_intervals([
        (
            segment.y_min,
            segment.y_max,
        )
        for segment in fitted_segments
    ])

    raw_coverage = sum(
        max(0.0, end - start)
        for start, end in intervals
    )

    coverage_ratio = (
        raw_coverage / vertical_span
        if vertical_span > 1.0
        else 0.0
    )

    occupied_bands = calculate_occupied_bands(
        fitted_segments,
        roi_y_min,
        roi_height,
    )

    support_count = len(
        fitted_segments
    )

    relative_start = (
        y_min - roi_y_min
    ) / max(roi_height, 1.0)

    relative_end = (
        y_max - roi_y_min
    ) / max(roi_height, 1.0)

    span_ratio = (
        vertical_span
        / max(roi_height, 1.0)
    )

    single_long_solid = (
        support_count >= 1
        and span_ratio
        >= SINGLE_LONG_LINE_SPAN_RATIO
    )

    normal_continuous_track = (
        support_count
        >= MIN_SUPPORT_SEGMENTS
        and span_ratio
        >= MIN_VERTICAL_SPAN_RATIO
        and occupied_bands
        >= MIN_OCCUPIED_BANDS
        and relative_start
        <= MAX_START_POSITION_RATIO
        and relative_end
        >= MIN_END_POSITION_RATIO
        and coverage_ratio
        >= MIN_RAW_COVERAGE_RATIO
    )

    if not (
        single_long_solid
        or normal_continuous_track
    ):
        return None

    score = (
        support_count * 1.4
        + occupied_bands * 2.0
        + span_ratio * 12.0
        + coverage_ratio * 5.0
        - residual_mean * 0.15
    )

    return LaneTrack(
        segments=fitted_segments,
        slope_xy=float(slope_xy),
        intercept_x=float(intercept_x),
        y_min=float(y_min),
        y_max=float(y_max),
        vertical_span=float(vertical_span),
        coverage_ratio=float(coverage_ratio),
        occupied_bands=int(occupied_bands),
        support_count=int(support_count),
        residual_mean=float(residual_mean),
        score=float(score),
    )


def tracks_are_duplicates(
    first: LaneTrack,
    second: LaneTrack,
    image_width: int,
) -> bool:
    overlap_start = max(
        first.y_min,
        second.y_min,
    )

    overlap_end = min(
        first.y_max,
        second.y_max,
    )

    if overlap_start > overlap_end:
        return False

    probe_ys = [
        overlap_start,
        (overlap_start + overlap_end) / 2.0,
        overlap_end,
    ]

    mean_distance = float(
        np.mean([
            abs(
                first.x_at(probe_y)
                - second.x_at(probe_y)
            )
            for probe_y in probe_ys
        ])
    )

    duplicate_distance = max(
        MIN_DUPLICATE_DISTANCE,
        image_width
        * DUPLICATE_DISTANCE_RATIO,
    )

    return (
        mean_distance
        <= duplicate_distance
        and abs(
            first.slope_xy
            - second.slope_xy
        )
        <= MAX_SLOPE_DIFFERENCE
    )


def deduplicate_tracks(
    tracks: List[LaneTrack],
    image_width: int,
):
    sorted_tracks = sorted(
        tracks,
        key=lambda item: item.score,
        reverse=True,
    )

    kept: List[LaneTrack] = []

    for candidate in sorted_tracks:
        duplicate = any(
            tracks_are_duplicates(
                candidate,
                existing,
                image_width,
            )
            for existing in kept
        )

        if not duplicate:
            kept.append(candidate)

    return kept


def find_spatial_lane_tracks(
    segments: List[LaneSegment],
    image_width: int,
    roi_y_min: float,
    roi_height: float,
):
    raw_clusters = cluster_segments(
        segments,
        image_width,
        roi_height,
    )

    valid_tracks = []

    for cluster in raw_clusters:
        track = build_lane_track(
            cluster,
            image_width,
            roi_y_min,
            roi_height,
        )

        if track is not None:
            valid_tracks.append(track)

    valid_tracks = deduplicate_tracks(
        valid_tracks,
        image_width,
    )

    bottom_y = (
        roi_y_min + roi_height
    )

    valid_tracks.sort(
        key=lambda track: track.x_at(
            bottom_y
        )
    )

    return valid_tracks, raw_clusters


# ============================================================
# 跨帧时间跟踪
# ============================================================

class TemporalLaneTracker:
    def __init__(self):
        self.tracks: List[TemporalTrack] = []
        self.next_track_id = 1

    def reset(self):
        self.tracks.clear()
        self.next_track_id = 1

    @staticmethod
    def _match_cost(
        temporal_track: TemporalTrack,
        detection: LaneTrack,
        probe_y: float,
    ) -> Tuple[float, float]:
        x_distance = abs(
            temporal_track.x_at(probe_y)
            - detection.x_at(probe_y)
        )

        slope_difference = abs(
            temporal_track.slope_xy
            - detection.slope_xy
        )

        return x_distance, slope_difference

    def update(
        self,
        detections: List[LaneTrack],
        image_width: int,
        probe_y: float,
    ):
        for track in self.tracks:
            track.matched_this_frame = False

        max_match_distance = max(
            TEMPORAL_MATCH_MIN_DISTANCE,
            image_width
            * TEMPORAL_MATCH_WIDTH_RATIO,
        )

        candidate_pairs = []

        for track_index, temporal_track in enumerate(
            self.tracks
        ):
            for detection_index, detection in enumerate(
                detections
            ):
                (
                    x_distance,
                    slope_difference,
                ) = self._match_cost(
                    temporal_track,
                    detection,
                    probe_y,
                )

                if (
                    x_distance
                    <= max_match_distance
                    and slope_difference
                    <= TEMPORAL_MAX_SLOPE_DIFFERENCE
                ):
                    cost = (
                        x_distance
                        + slope_difference * 80.0
                    )

                    candidate_pairs.append(
                        (
                            cost,
                            track_index,
                            detection_index,
                        )
                    )

        candidate_pairs.sort(
            key=lambda item: item[0]
        )

        used_track_indexes = set()
        used_detection_indexes = set()

        for (
            _,
            track_index,
            detection_index,
        ) in candidate_pairs:
            if (
                track_index in used_track_indexes
                or detection_index
                in used_detection_indexes
            ):
                continue

            temporal_track = self.tracks[
                track_index
            ]

            detection = detections[
                detection_index
            ]

            alpha = TEMPORAL_EMA_ALPHA

            temporal_track.slope_xy = (
                (1.0 - alpha)
                * temporal_track.slope_xy
                + alpha
                * detection.slope_xy
            )

            temporal_track.intercept_x = (
                (1.0 - alpha)
                * temporal_track.intercept_x
                + alpha
                * detection.intercept_x
            )

            temporal_track.y_min = (
                (1.0 - alpha)
                * temporal_track.y_min
                + alpha
                * detection.y_min
            )

            temporal_track.y_max = (
                (1.0 - alpha)
                * temporal_track.y_max
                + alpha
                * detection.y_max
            )

            temporal_track.score = (
                (1.0 - alpha)
                * temporal_track.score
                + alpha
                * detection.score
            )

            temporal_track.support_count = (
                detection.support_count
            )

            temporal_track.occupied_bands = (
                detection.occupied_bands
            )

            temporal_track.hits += 1
            temporal_track.missed = 0
            temporal_track.age += 1
            temporal_track.matched_this_frame = True

            used_track_indexes.add(
                track_index
            )

            used_detection_indexes.add(
                detection_index
            )

        # 未匹配的旧轨迹先保留几帧，处理短暂遮挡。
        for track_index, temporal_track in enumerate(
            self.tracks
        ):
            if track_index not in used_track_indexes:
                temporal_track.missed += 1
                temporal_track.age += 1
                temporal_track.matched_this_frame = False

        # 未匹配的新检测创建新轨迹。
        for detection_index, detection in enumerate(
            detections
        ):
            if detection_index in used_detection_indexes:
                continue

            self.tracks.append(
                TemporalTrack(
                    track_id=self.next_track_id,
                    slope_xy=detection.slope_xy,
                    intercept_x=detection.intercept_x,
                    y_min=detection.y_min,
                    y_max=detection.y_max,
                    score=detection.score,
                    support_count=detection.support_count,
                    occupied_bands=detection.occupied_bands,
                )
            )

            self.next_track_id += 1

        # 删除长时间丢失的轨迹。
        self.tracks = [
            track
            for track in self.tracks
            if track.missed
            <= TEMPORAL_MAX_MISSED_FRAMES
        ]

        # 按画面下端横坐标排序。
        self.tracks.sort(
            key=lambda track: track.x_at(
                probe_y
            )
        )

    def confirmed_tracks(self):
        return [
            track
            for track in self.tracks
            if track.hits
            >= TEMPORAL_MIN_CONFIRM_HITS
        ]


# ============================================================
# 单帧检测
# ============================================================

def detect_lanes_in_frame(
    frame,
    roi_points: np.ndarray,
):
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        GAUSSIAN_KERNEL,
        0,
    )

    edges = cv2.Canny(
        blurred,
        CANNY_LOW,
        CANNY_HIGH,
    )

    mask = np.zeros_like(edges)

    cv2.fillPoly(
        mask,
        [
            roi_points.reshape(
                (-1, 1, 2)
            )
        ],
        255,
    )

    masked_edges = cv2.bitwise_and(
        edges,
        mask,
    )

    raw_lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    initial_segments = extract_initial_segments(
        raw_lines
    )

    roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(
        roi_points.reshape(
            (-1, 1, 2)
        )
    )

    (
        spatial_tracks,
        raw_clusters,
    ) = find_spatial_lane_tracks(
        initial_segments,
        image_width=frame.shape[1],
        roi_y_min=float(roi_y),
        roi_height=float(roi_h),
    )

    return {
        "gray": gray,
        "blurred": blurred,
        "edges": edges,
        "masked_edges": masked_edges,
        "raw_lines": raw_lines,
        "initial_segments": initial_segments,
        "raw_clusters": raw_clusters,
        "spatial_tracks": spatial_tracks,
        "roi_rect": (
            roi_x,
            roi_y,
            roi_w,
            roi_h,
        ),
    }


# ============================================================
# 绘制实时结果
# ============================================================

TRACK_COLORS = [
    (40, 255, 110),
    (255, 180, 40),
    (220, 80, 255),
    (40, 220, 255),
    (255, 100, 80),
    (160, 255, 60),
    (255, 80, 180),
    (80, 160, 255),
]


def draw_roi_overlay(
    frame,
    roi_points,
):
    canvas = frame.copy()
    polygon = roi_points.reshape(
        (-1, 1, 2)
    )

    layer = canvas.copy()

    cv2.fillPoly(
        layer,
        [polygon],
        (40, 180, 220),
    )

    canvas = cv2.addWeighted(
        layer,
        0.08,
        canvas,
        0.92,
        0,
    )

    cv2.polylines(
        canvas,
        [polygon],
        True,
        (0, 225, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas


def draw_temporal_tracks(
    frame,
    temporal_tracks: List[TemporalTrack],
    roi_points: np.ndarray,
):
    canvas = draw_roi_overlay(
        frame,
        roi_points,
    )

    h, w = canvas.shape[:2]

    for index, track in enumerate(
        temporal_tracks
    ):
        color = TRACK_COLORS[
            index % len(TRACK_COLORS)
        ]

        y_top = int(round(track.y_min))
        y_bottom = int(round(track.y_max))

        y_top = max(
            0,
            min(h - 1, y_top),
        )

        y_bottom = max(
            0,
            min(h - 1, y_bottom),
        )

        x_top = int(round(
            track.x_at(y_top)
        ))

        x_bottom = int(round(
            track.x_at(y_bottom)
        ))

        x_top = max(
            0,
            min(w - 1, x_top),
        )

        x_bottom = max(
            0,
            min(w - 1, x_bottom),
        )

        # 当前帧有匹配时使用实线感较强的粗线；
        # 短暂丢失时画得稍细，表示由时间跟踪保持。
        thickness = (
            7
            if track.matched_this_frame
            else 4
        )

        cv2.line(
            canvas,
            (x_top, y_top),
            (x_bottom, y_bottom),
            color,
            thickness,
            cv2.LINE_AA,
        )

        label = (
            f"L{track.track_id} "
            f"H:{track.hits} "
            f"M:{track.missed}"
        )

        label_x = max(
            5,
            min(
                w - 170,
                x_top + 8,
            ),
        )

        label_y = max(
            25,
            min(
                h - 10,
                y_top + 25,
            ),
        )

        cv2.putText(
            canvas,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )

    return canvas


def draw_masked_edges_panel(
    masked_edges,
    initial_segments: List[LaneSegment],
):
    canvas = cv2.cvtColor(
        masked_edges,
        cv2.COLOR_GRAY2BGR,
    )

    for segment in initial_segments:
        cv2.line(
            canvas,
            (segment.x1, segment.y1),
            (segment.x2, segment.y2),
            (255, 160, 30),
            2,
            cv2.LINE_AA,
        )

    return canvas


def draw_status_bar(
    image,
    fps_value: float,
    frame_index: int,
    total_frames: int,
    initial_count: int,
    spatial_count: int,
    temporal_count: int,
    paused: bool,
):
    canvas = image.copy()

    overlay_height = 54

    cv2.rectangle(
        canvas,
        (0, 0),
        (canvas.shape[1], overlay_height),
        (10, 16, 24),
        -1,
    )

    playback_text = (
        "PAUSED"
        if paused
        else "PLAYING"
    )

    if total_frames > 0:
        progress_text = (
            f"{frame_index}/{total_frames}"
        )
    else:
        progress_text = str(frame_index)

    line1 = (
        f"{playback_text} | FPS:{fps_value:.1f} | "
        f"Frame:{progress_text}"
    )

    line2 = (
        f"Hough vertical:{initial_count} | "
        f"Spatial lanes:{spatial_count} | "
        f"Stable lanes:{temporal_count}"
    )

    cv2.putText(
        canvas,
        line1,
        (14, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        (240, 245, 250),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        line2,
        (14, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (70, 220, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas


def build_display(
    result_frame,
    edge_panel,
    fps_value: float,
    frame_index: int,
    total_frames: int,
    initial_count: int,
    spatial_count: int,
    temporal_count: int,
    paused: bool,
):
    left = add_title(
        result_frame,
        "Continuous Lane Detection Result",
    )

    right = add_title(
        edge_panel,
        "ROI Edges + Near-vertical Hough Segments",
    )

    panel_height = max(
        left.shape[0],
        right.shape[0],
    )

    panel_width = max(
        left.shape[1],
        right.shape[1],
    )

    left = cv2.resize(
        left,
        (panel_width, panel_height),
        interpolation=cv2.INTER_AREA,
    )

    right = cv2.resize(
        right,
        (panel_width, panel_height),
        interpolation=cv2.INTER_AREA,
    )

    separator = np.full(
        (
            panel_height,
            6,
            3,
        ),
        70,
        dtype=np.uint8,
    )

    combined = np.hstack([
        left,
        separator,
        right,
    ])

    combined = draw_status_bar(
        combined,
        fps_value=fps_value,
        frame_index=frame_index,
        total_frames=total_frames,
        initial_count=initial_count,
        spatial_count=spatial_count,
        temporal_count=temporal_count,
        paused=paused,
    )

    combined, _ = resize_to_fit(
        combined,
        MAX_DISPLAY_WIDTH,
        MAX_DISPLAY_HEIGHT,
    )

    return combined


# ============================================================
# 主程序
# ============================================================

def main():
    video_path = choose_video()

    if not video_path:
        print("未选择视频，程序退出。")
        return

    capture = cv2.VideoCapture(
        video_path
    )

    if not capture.isOpened():
        print(f"[错误] 无法打开视频：{video_path}")
        return

    source_fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    if (
        not math.isfinite(source_fps)
        or source_fps <= 1.0
    ):
        source_fps = 25.0

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    success, first_source_frame = capture.read()

    if not success or first_source_frame is None:
        capture.release()
        print("无法读取视频第一帧。")
        return

    first_frame, process_scale = resize_to_max_width(
        first_source_frame,
        PROCESS_MAX_WIDTH,
    )

    print("=" * 82)
    print("本地视频已打开。")
    print(f"视频路径：{Path(video_path)}")
    print(f"原始 FPS：{source_fps:.2f}")
    print(f"总帧数：{total_frames}")
    print(
        f"处理尺寸：{first_frame.shape[1]} x "
        f"{first_frame.shape[0]}"
    )
    print("请在第一帧中圈出道路 ROI，然后按 Enter。")
    print("=" * 82)

    roi_points = ROISelector(
        first_frame
    ).select()

    if roi_points is None:
        capture.release()
        cv2.destroyAllWindows()
        print("已取消 ROI 选择。")
        return

    # 回到第一帧开始播放。
    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        0,
    )

    temporal_tracker = TemporalLaneTracker()

    paused = False
    single_step = False
    frame_index = 0
    processed_counter = 0

    last_detection = None
    last_source_frame = None
    last_processed_frame = None

    fps_ema = 0.0
    last_loop_time = time.perf_counter()

    cv2.namedWindow(
        VIDEO_WINDOW,
        cv2.WINDOW_NORMAL,
    )

    while True:
        should_read_frame = (
            not paused
            or single_step
            or last_source_frame is None
        )

        if should_read_frame:
            success, source_frame = capture.read()

            if not success or source_frame is None:
                print("视频播放结束。")
                break

            frame_index = int(
                capture.get(
                    cv2.CAP_PROP_POS_FRAMES
                )
            )

            processed_frame, _ = resize_to_max_width(
                source_frame,
                PROCESS_MAX_WIDTH,
            )

            last_source_frame = source_frame
            last_processed_frame = processed_frame

            processed_counter += 1

            if (
                processed_counter
                % PROCESS_EVERY_N_FRAMES
                == 0
                or last_detection is None
            ):
                detection_start = time.perf_counter()

                detection = detect_lanes_in_frame(
                    processed_frame,
                    roi_points,
                )

                roi_x, roi_y, roi_w, roi_h = detection[
                    "roi_rect"
                ]

                temporal_tracker.update(
                    detection["spatial_tracks"],
                    image_width=processed_frame.shape[1],
                    probe_y=float(
                        roi_y + roi_h
                    ),
                )

                last_detection = detection

                detection_elapsed = (
                    time.perf_counter()
                    - detection_start
                )

                current_processing_fps = (
                    1.0 / detection_elapsed
                    if detection_elapsed > 0
                    else 0.0
                )

                if fps_ema <= 0:
                    fps_ema = current_processing_fps
                else:
                    fps_ema = (
                        0.90 * fps_ema
                        + 0.10
                        * current_processing_fps
                    )

            single_step = False

        if (
            last_processed_frame is None
            or last_detection is None
        ):
            continue

        stable_tracks = (
            temporal_tracker.confirmed_tracks()
        )

        result_frame = draw_temporal_tracks(
            last_processed_frame,
            stable_tracks,
            roi_points,
        )

        edge_panel = draw_masked_edges_panel(
            last_detection["masked_edges"],
            last_detection["initial_segments"],
        )

        display = build_display(
            result_frame,
            edge_panel,
            fps_value=fps_ema,
            frame_index=frame_index,
            total_frames=total_frames,
            initial_count=len(
                last_detection[
                    "initial_segments"
                ]
            ),
            spatial_count=len(
                last_detection[
                    "spatial_tracks"
                ]
            ),
            temporal_count=len(
                stable_tracks
            ),
            paused=paused,
        )

        cv2.imshow(
            VIDEO_WINDOW,
            display,
        )

        # 尽量按照原视频帧率播放。
        target_delay_ms = max(
            1,
            int(round(
                1000.0 / source_fps
            )),
        )

        key = cv2.waitKey(
            target_delay_ms
            if not paused
            else 30
        )

        if key == -1:
            key_code = -1
        else:
            key_code = key & 0xFF

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

        elif key_code in (
            ord("r"),
            ord("R"),
        ):
            # 在当前处理帧重新选择 ROI。
            new_roi = ROISelector(
                last_processed_frame
            ).select()

            if new_roi is not None:
                roi_points = new_roi
                temporal_tracker.reset()
                last_detection = None
                print("ROI 已重新选择，时间轨迹已重置。")

        # OpenCV 在 Windows 中方向键通常返回扩展键码，
        # 这里同时兼容常见低 8 位值。
        elif key in (
            2555904,
            83,
        ):
            current_position = int(
                capture.get(
                    cv2.CAP_PROP_POS_FRAMES
                )
            )

            jump_frames = int(
                round(source_fps * 5.0)
            )

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                min(
                    max(total_frames - 1, 0),
                    current_position
                    + jump_frames,
                ),
            )

            temporal_tracker.reset()
            last_detection = None

        elif key in (
            2424832,
            81,
        ):
            current_position = int(
                capture.get(
                    cv2.CAP_PROP_POS_FRAMES
                )
            )

            jump_frames = int(
                round(source_fps * 5.0)
            )

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                max(
                    0,
                    current_position
                    - jump_frames,
                ),
            )

            temporal_tracker.reset()
            last_detection = None

        try:
            if cv2.getWindowProperty(
                VIDEO_WINDOW,
                cv2.WND_PROP_VISIBLE,
            ) < 1:
                break
        except cv2.error:
            break

        now = time.perf_counter()
        last_loop_time = now

    capture.release()
    cv2.destroyAllWindows()

    print("=" * 82)
    print("视频检测已结束。")
    print("程序未保存任何视频或图片。")
    print("=" * 82)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        sys.exit(0)
    except Exception as exc:
        cv2.destroyAllWindows()
        print(f"[程序异常] {type(exc).__name__}: {exc}")
        raise
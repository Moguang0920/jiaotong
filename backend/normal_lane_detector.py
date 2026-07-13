# -*- coding: utf-8 -*-
"""正常道路模式专用的连续车道线检测器。

该模块没有独立线程，也不会在车牌、热力图或禁停模式中运行。
只有 plate_runtime_backend.py 当前 detector_model == "normal" 且用户已经
提交正常道路 ROI 时，YOLO 推理线程才会调用本模块。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# 图像预处理
GAUSSIAN_KERNEL = (5, 5)
CANNY_LOW = 50
CANNY_HIGH = 150

# HoughLinesP
HOUGH_THRESHOLD = 32
HOUGH_MIN_LINE_LENGTH = 24
HOUGH_MAX_LINE_GAP = 16
MAX_INITIAL_SEGMENTS = 140

# 近竖直线段初筛。角度相对水平线，90 度为竖直。
INITIAL_MIN_LENGTH = 30.0
INITIAL_MIN_ANGLE = 56.0
INITIAL_MAX_ANGLE = 90.0

# 单帧空间连续性
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
SINGLE_LONG_LINE_SPAN_RATIO = 0.62
MIN_DUPLICATE_DISTANCE = 14.0
DUPLICATE_DISTANCE_RATIO = 0.014

# 跨帧稳定
TEMPORAL_MATCH_MIN_DISTANCE = 34.0
TEMPORAL_MATCH_WIDTH_RATIO = 0.035
TEMPORAL_MAX_SLOPE_DIFFERENCE = 0.30
TEMPORAL_EMA_ALPHA = 0.32
TEMPORAL_MIN_CONFIRM_HITS = 2
TEMPORAL_MAX_MISSED_FRAMES = 6


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


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, first: int, second: int) -> None:
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


class TemporalLaneTracker:
    def __init__(self) -> None:
        self.tracks: List[TemporalTrack] = []
        self.next_track_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self.next_track_id = 1

    def update(self, detections: List[LaneTrack], image_width: int, probe_y: float) -> None:
        for track in self.tracks:
            track.matched_this_frame = False

        max_match_distance = max(
            TEMPORAL_MATCH_MIN_DISTANCE,
            image_width * TEMPORAL_MATCH_WIDTH_RATIO,
        )
        candidate_pairs: List[Tuple[float, int, int]] = []

        for track_index, old in enumerate(self.tracks):
            for detection_index, detection in enumerate(detections):
                x_distance = abs(old.x_at(probe_y) - detection.x_at(probe_y))
                slope_difference = abs(old.slope_xy - detection.slope_xy)
                if (
                    x_distance <= max_match_distance
                    and slope_difference <= TEMPORAL_MAX_SLOPE_DIFFERENCE
                ):
                    candidate_pairs.append(
                        (x_distance + slope_difference * 80.0, track_index, detection_index)
                    )

        candidate_pairs.sort(key=lambda item: item[0])
        used_tracks = set()
        used_detections = set()

        for _, track_index, detection_index in candidate_pairs:
            if track_index in used_tracks or detection_index in used_detections:
                continue
            old = self.tracks[track_index]
            detection = detections[detection_index]
            alpha = TEMPORAL_EMA_ALPHA
            old.slope_xy = (1.0 - alpha) * old.slope_xy + alpha * detection.slope_xy
            old.intercept_x = (1.0 - alpha) * old.intercept_x + alpha * detection.intercept_x
            old.y_min = (1.0 - alpha) * old.y_min + alpha * detection.y_min
            old.y_max = (1.0 - alpha) * old.y_max + alpha * detection.y_max
            old.score = (1.0 - alpha) * old.score + alpha * detection.score
            old.support_count = detection.support_count
            old.occupied_bands = detection.occupied_bands
            old.hits += 1
            old.missed = 0
            old.age += 1
            old.matched_this_frame = True
            used_tracks.add(track_index)
            used_detections.add(detection_index)

        for track_index, old in enumerate(self.tracks):
            if track_index not in used_tracks:
                old.missed += 1
                old.age += 1
                old.matched_this_frame = False

        for detection_index, detection in enumerate(detections):
            if detection_index in used_detections:
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

        self.tracks = [
            track for track in self.tracks if track.missed <= TEMPORAL_MAX_MISSED_FRAMES
        ]
        self.tracks.sort(key=lambda track: track.x_at(probe_y))

    def confirmed_tracks(self) -> List[TemporalTrack]:
        return [track for track in self.tracks if track.hits >= TEMPORAL_MIN_CONFIRM_HITS]


def sanitize_normalized_roi(points: Sequence[Sequence[float]]) -> List[List[float]]:
    """校验并归一化前端提交的多边形点。"""
    cleaned: List[List[float]] = []
    for point in points or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            x = max(0.0, min(1.0, float(point[0])))
            y = max(0.0, min(1.0, float(point[1])))
        except Exception:
            continue
        cleaned.append([round(x, 7), round(y, 7)])

    if len(cleaned) < 3:
        return []

    # 鞋带公式，过滤几乎没有面积的无效多边形。
    area = 0.0
    for index, point in enumerate(cleaned):
        next_point = cleaned[(index + 1) % len(cleaned)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    if abs(area) * 0.5 < 0.001:
        return []
    return cleaned[:32]


def normalized_roi_to_pixels(
    normalized_points: Sequence[Sequence[float]], width: int, height: int
) -> np.ndarray:
    points = []
    for x_norm, y_norm in normalized_points:
        x = int(round(float(x_norm) * max(width - 1, 1)))
        y = int(round(float(y_norm) * max(height - 1, 1)))
        points.append([max(0, min(width - 1, x)), max(0, min(height - 1, y))])
    return np.asarray(points, dtype=np.int32)


def _create_segment(x1: int, y1: int, x2: int, y2: int) -> Optional[LaneSegment]:
    if y1 > y2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    if abs(dy) < 1.0:
        return None
    length = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(abs(dy), abs(dx) + 1e-9))
    if length < INITIAL_MIN_LENGTH or not (INITIAL_MIN_ANGLE <= angle <= INITIAL_MAX_ANGLE):
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


def _extract_initial_segments(raw_lines: Any) -> List[LaneSegment]:
    segments: List[LaneSegment] = []
    if raw_lines is None:
        return segments
    for raw in raw_lines:
        x1, y1, x2, y2 = map(int, raw[0])
        segment = _create_segment(x1, y1, x2, y2)
        if segment is not None:
            segments.append(segment)
    segments.sort(key=lambda item: item.length, reverse=True)
    return segments[:MAX_INITIAL_SEGMENTS]


def _vertical_gap(first: LaneSegment, second: LaneSegment) -> float:
    if first.y_max < second.y_min:
        return second.y_min - first.y_max
    if second.y_max < first.y_min:
        return first.y_min - second.y_max
    return 0.0


def _lateral_distance(first: LaneSegment, second: LaneSegment) -> float:
    overlap_start = max(first.y_min, second.y_min)
    overlap_end = min(first.y_max, second.y_max)
    if overlap_start <= overlap_end:
        probe_y = (overlap_start + overlap_end) / 2.0
    elif first.y_max < second.y_min:
        probe_y = (first.y_max + second.y_min) / 2.0
    else:
        probe_y = (second.y_max + first.y_min) / 2.0
    return abs(first.x_at(probe_y) - second.x_at(probe_y))


def _segments_are_continuous(
    first: LaneSegment,
    second: LaneSegment,
    max_lateral_distance: float,
    max_vertical_gap: float,
) -> bool:
    if abs(first.angle - second.angle) > MAX_ANGLE_DIFFERENCE:
        return False
    if abs(first.slope_xy - second.slope_xy) > MAX_SLOPE_DIFFERENCE:
        return False
    gap = _vertical_gap(first, second)
    if gap > max_vertical_gap:
        return False
    gap_bonus = min(max_lateral_distance * 0.35, gap * 0.08)
    return _lateral_distance(first, second) <= max_lateral_distance + gap_bonus


def _cluster_segments(
    segments: List[LaneSegment], image_width: int, roi_height: float
) -> List[List[LaneSegment]]:
    if not segments:
        return []
    max_lateral_distance = max(MIN_LATERAL_DISTANCE, image_width * LATERAL_DISTANCE_RATIO)
    max_vertical_gap = max(MIN_VERTICAL_GAP, roi_height * VERTICAL_GAP_RATIO)
    union_find = UnionFind(len(segments))
    for first_index in range(len(segments)):
        for second_index in range(first_index + 1, len(segments)):
            if _segments_are_continuous(
                segments[first_index],
                segments[second_index],
                max_lateral_distance,
                max_vertical_gap,
            ):
                union_find.union(first_index, second_index)
    grouped: Dict[int, List[LaneSegment]] = {}
    for index, segment in enumerate(segments):
        grouped.setdefault(union_find.find(index), []).append(segment)
    return list(grouped.values())


def _weighted_fit_x_from_y(segments: List[LaneSegment]) -> Tuple[float, float]:
    y_values: List[float] = []
    x_values: List[float] = []
    weights: List[float] = []
    for segment in segments:
        weight = math.sqrt(max(segment.length, 1.0))
        y_values.extend([float(segment.y1), float(segment.y2)])
        x_values.extend([float(segment.x1), float(segment.x2)])
        weights.extend([weight, weight])
    coefficients = np.polyfit(
        np.asarray(y_values, dtype=np.float64),
        np.asarray(x_values, dtype=np.float64),
        deg=1,
        w=np.asarray(weights, dtype=np.float64),
    )
    return float(coefficients[0]), float(coefficients[1])


def _segment_fit_residual(
    segment: LaneSegment, slope_xy: float, intercept_x: float
) -> float:
    probe_y_values = [float(segment.y1), (segment.y_min + segment.y_max) / 2.0, float(segment.y2)]
    residuals = []
    for probe_y in probe_y_values:
        predicted_x = slope_xy * probe_y + intercept_x
        residuals.append(abs(segment.x_at(probe_y) - predicted_x))
    return float(np.mean(residuals))


def _robust_fit_cluster(
    segments: List[LaneSegment], image_width: int
) -> Tuple[List[LaneSegment], float, float, float]:
    working = list(segments)
    residual_limit = max(MIN_FIT_RESIDUAL, image_width * FIT_RESIDUAL_WIDTH_RATIO)
    for _ in range(3):
        if not working:
            break
        slope_xy, intercept_x = _weighted_fit_x_from_y(working)
        filtered = [
            segment
            for segment in working
            if _segment_fit_residual(segment, slope_xy, intercept_x) <= residual_limit
        ]
        if len(filtered) == len(working):
            working = filtered
            break
        if not filtered:
            break
        working = filtered
    if not working:
        return [], 0.0, 0.0, float("inf")
    slope_xy, intercept_x = _weighted_fit_x_from_y(working)
    residual_mean = float(
        np.mean([
            _segment_fit_residual(segment, slope_xy, intercept_x)
            for segment in working
        ])
    )
    return working, slope_xy, intercept_x, residual_mean


def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda item: item[0])
    merged: List[List[float]] = [[float(sorted_intervals[0][0]), float(sorted_intervals[0][1])]]
    for start, end in sorted_intervals[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], float(end))
        else:
            merged.append([float(start), float(end)])
    return [(start, end) for start, end in merged]


def _calculate_occupied_bands(
    segments: List[LaneSegment], roi_y_min: float, roi_height: float
) -> int:
    if roi_height <= 0:
        return 0
    band_height = roi_height / VERTICAL_BAND_COUNT
    occupied = set()
    for segment in segments:
        start_index = int(math.floor((segment.y_min - roi_y_min) / band_height))
        end_index = int(math.floor((segment.y_max - roi_y_min) / band_height))
        start_index = max(0, min(VERTICAL_BAND_COUNT - 1, start_index))
        end_index = max(0, min(VERTICAL_BAND_COUNT - 1, end_index))
        for band_index in range(start_index, end_index + 1):
            occupied.add(band_index)
    return len(occupied)


def _build_lane_track(
    raw_cluster: List[LaneSegment],
    image_width: int,
    roi_y_min: float,
    roi_height: float,
) -> Optional[LaneTrack]:
    fitted_segments, slope_xy, intercept_x, residual_mean = _robust_fit_cluster(
        raw_cluster, image_width
    )
    if not fitted_segments:
        return None
    y_min = min(segment.y_min for segment in fitted_segments)
    y_max = max(segment.y_max for segment in fitted_segments)
    vertical_span = max(0.0, y_max - y_min)
    intervals = _merge_intervals([(segment.y_min, segment.y_max) for segment in fitted_segments])
    raw_coverage = sum(max(0.0, end - start) for start, end in intervals)
    coverage_ratio = raw_coverage / vertical_span if vertical_span > 1.0 else 0.0
    occupied_bands = _calculate_occupied_bands(fitted_segments, roi_y_min, roi_height)
    support_count = len(fitted_segments)
    relative_start = (y_min - roi_y_min) / max(roi_height, 1.0)
    relative_end = (y_max - roi_y_min) / max(roi_height, 1.0)
    span_ratio = vertical_span / max(roi_height, 1.0)
    single_long_solid = support_count >= 1 and span_ratio >= SINGLE_LONG_LINE_SPAN_RATIO
    normal_continuous_track = (
        support_count >= MIN_SUPPORT_SEGMENTS
        and span_ratio >= MIN_VERTICAL_SPAN_RATIO
        and occupied_bands >= MIN_OCCUPIED_BANDS
        and relative_start <= MAX_START_POSITION_RATIO
        and relative_end >= MIN_END_POSITION_RATIO
        and coverage_ratio >= MIN_RAW_COVERAGE_RATIO
    )
    if not (single_long_solid or normal_continuous_track):
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


def _tracks_are_duplicates(first: LaneTrack, second: LaneTrack, image_width: int) -> bool:
    overlap_start = max(first.y_min, second.y_min)
    overlap_end = min(first.y_max, second.y_max)
    if overlap_start > overlap_end:
        return False
    probe_ys = [overlap_start, (overlap_start + overlap_end) / 2.0, overlap_end]
    mean_distance = float(
        np.mean([abs(first.x_at(probe_y) - second.x_at(probe_y)) for probe_y in probe_ys])
    )
    duplicate_distance = max(MIN_DUPLICATE_DISTANCE, image_width * DUPLICATE_DISTANCE_RATIO)
    return (
        mean_distance <= duplicate_distance
        and abs(first.slope_xy - second.slope_xy) <= MAX_SLOPE_DIFFERENCE
    )


def _find_spatial_lane_tracks(
    segments: List[LaneSegment], image_width: int, roi_y_min: float, roi_height: float
) -> Tuple[List[LaneTrack], List[List[LaneSegment]]]:
    clusters = _cluster_segments(segments, image_width, roi_height)
    candidates: List[LaneTrack] = []
    for cluster in clusters:
        track = _build_lane_track(cluster, image_width, roi_y_min, roi_height)
        if track is not None:
            candidates.append(track)
    candidates.sort(key=lambda item: item.score, reverse=True)
    kept: List[LaneTrack] = []
    for candidate in candidates:
        if not any(_tracks_are_duplicates(candidate, existing, image_width) for existing in kept):
            kept.append(candidate)
    bottom_y = roi_y_min + roi_height
    kept.sort(key=lambda item: item.x_at(bottom_y))
    return kept, clusters


def _extend_track_to_roi(
    track: TemporalTrack,
    roi_polygon: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> Tuple[int, int, int, int, bool]:
    """把稳定轨迹沿拟合方向延长到用户所选 ROI 的真实上下边界。

    旧版直接使用 Hough 支撑线段的 y_min/y_max，因此最终绘制线会比道路区域短。
    这里逐行计算拟合直线 x(y)，并用 pointPolygonTest 判断该点是否位于 ROI 内，
    最终选择与原始支撑区间重叠最多的连续区间作为绘制端点。这样既能延长，
    又不会越过用户圈选的道路多边形。
    """
    if roi_polygon is None or len(roi_polygon) < 3:
        y_top = max(0, min(frame_height - 1, int(round(track.y_min))))
        y_bottom = max(0, min(frame_height - 1, int(round(track.y_max))))
        x_top = max(0, min(frame_width - 1, int(round(track.x_at(y_top)))))
        x_bottom = max(0, min(frame_width - 1, int(round(track.x_at(y_bottom)))))
        return x_top, y_top, x_bottom, y_bottom, False

    roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(roi_polygon)
    start_y = max(0, int(roi_y))
    end_y = min(frame_height - 1, int(roi_y + roi_h - 1))

    valid_rows: List[Tuple[int, int]] = []
    for y in range(start_y, end_y + 1):
        x = int(round(track.x_at(float(y))))
        if x < 0 or x >= frame_width:
            continue
        inside = cv2.pointPolygonTest(
            roi_polygon,
            (float(x), float(y)),
            False,
        )
        if inside >= 0:
            valid_rows.append((y, x))

    if not valid_rows:
        y_top = max(0, min(frame_height - 1, int(round(track.y_min))))
        y_bottom = max(0, min(frame_height - 1, int(round(track.y_max))))
        x_top = max(0, min(frame_width - 1, int(round(track.x_at(y_top)))))
        x_bottom = max(0, min(frame_width - 1, int(round(track.x_at(y_bottom)))))
        return x_top, y_top, x_bottom, y_bottom, False

    # 将有效行拆成连续区间。对于凹多边形，直线可能多次进出 ROI。
    runs: List[List[Tuple[int, int]]] = []
    current: List[Tuple[int, int]] = []
    previous_y: Optional[int] = None
    for item in valid_rows:
        y, _ = item
        if previous_y is None or y == previous_y + 1:
            current.append(item)
        else:
            if current:
                runs.append(current)
            current = [item]
        previous_y = y
    if current:
        runs.append(current)

    support_top = float(min(track.y_min, track.y_max))
    support_bottom = float(max(track.y_min, track.y_max))

    def run_score(run: List[Tuple[int, int]]) -> Tuple[float, int]:
        run_top = float(run[0][0])
        run_bottom = float(run[-1][0])
        overlap = max(
            0.0,
            min(run_bottom, support_bottom) - max(run_top, support_top),
        )
        return overlap, len(run)

    best_run = max(runs, key=run_score)
    y_top, x_top = best_run[0]
    y_bottom, x_bottom = best_run[-1]

    return (
        max(0, min(frame_width - 1, int(x_top))),
        max(0, min(frame_height - 1, int(y_top))),
        max(0, min(frame_width - 1, int(x_bottom))),
        max(0, min(frame_height - 1, int(y_bottom))),
        True,
    )


class NormalLaneDetector:
    """仅供 normal.onnx 模式调用的有状态检测器。"""

    def __init__(self) -> None:
        self.roi_normalized: List[List[float]] = []
        self.temporal_tracker = TemporalLaneTracker()

    def configure(self, normalized_points: Sequence[Sequence[float]]) -> bool:
        cleaned = sanitize_normalized_roi(normalized_points)
        self.roi_normalized = cleaned
        self.temporal_tracker.reset()
        return bool(cleaned)

    def reset(self) -> None:
        self.roi_normalized = []
        self.temporal_tracker.reset()

    @property
    def configured(self) -> bool:
        return len(self.roi_normalized) >= 3

    def process(self, frame: np.ndarray, frame_id: int = 0) -> Dict[str, Any]:
        started = time.perf_counter()
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("正常道路车道线检测收到空帧")
        if not self.configured:
            return self.empty_result(
                frame_id=frame_id,
                frame_width=int(frame.shape[1]),
                frame_height=int(frame.shape[0]),
                status="waiting_roi",
                message="请先选择正常道路 ROI。",
            )

        height, width = frame.shape[:2]
        roi_points = normalized_roi_to_pixels(self.roi_normalized, width, height)
        roi_polygon = roi_points.reshape((-1, 1, 2))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)
        edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [roi_polygon], 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        raw_lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LINE_LENGTH,
            maxLineGap=HOUGH_MAX_LINE_GAP,
        )
        initial_segments = _extract_initial_segments(raw_lines)
        roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(roi_polygon)
        spatial_tracks, clusters = _find_spatial_lane_tracks(
            initial_segments,
            image_width=width,
            roi_y_min=float(roi_y),
            roi_height=float(roi_h),
        )
        self.temporal_tracker.update(
            spatial_tracks,
            image_width=width,
            probe_y=float(roi_y + roi_h),
        )
        stable_tracks = self.temporal_tracker.confirmed_tracks()

        lane_lines: List[Dict[str, Any]] = []
        for track in stable_tracks:
            # 支撑范围用于诊断；实际绘制端点延长到 ROI 多边形边界。
            support_y_top = max(0, min(height - 1, int(round(track.y_min))))
            support_y_bottom = max(0, min(height - 1, int(round(track.y_max))))
            support_x_top = max(0, min(width - 1, int(round(track.x_at(support_y_top)))))
            support_x_bottom = max(0, min(width - 1, int(round(track.x_at(support_y_bottom)))))
            x_top, y_top, x_bottom, y_bottom, extended = _extend_track_to_roi(
                track,
                roi_polygon,
                width,
                height,
            )
            lane_lines.append({
                "track_id": int(track.track_id),
                "x1": x_top,
                "y1": y_top,
                "x2": x_bottom,
                "y2": y_bottom,
                "support_x1": support_x_top,
                "support_y1": support_y_top,
                "support_x2": support_x_bottom,
                "support_y2": support_y_bottom,
                "extended_to_roi": bool(extended),
                "hits": int(track.hits),
                "missed": int(track.missed),
                "matched": bool(track.matched_this_frame),
                "support_count": int(track.support_count),
                "occupied_bands": int(track.occupied_bands),
                "score": round(float(track.score), 3),
            })

        candidate_lines = [
            {
                "x1": int(segment.x1),
                "y1": int(segment.y1),
                "x2": int(segment.x2),
                "y2": int(segment.y2),
                "length": round(float(segment.length), 2),
                "angle": round(float(segment.angle), 2),
            }
            for segment in initial_segments[:120]
        ]

        status = "ok" if lane_lines else "warming_up"
        if lane_lines:
            message = f"正常道路模式：已稳定绘制 {len(lane_lines)} 条连续车道线。"
        elif spatial_tracks:
            message = "已发现空间连续车道线，正在进行跨帧稳定确认。"
        else:
            message = "当前 ROI 内暂未形成满足连续性要求的车道线。"

        return {
            "enabled": True,
            "mode": "normal_only",
            "frame_id": int(frame_id),
            "updated_at": time.time(),
            "status": status,
            "progress": 100,
            "roi": [[int(x), int(y)] for x, y in roi_points.tolist()],
            "roi_normalized": [list(point) for point in self.roi_normalized],
            "candidate_lines": candidate_lines,
            "lane_lines": lane_lines,
            "stable_lane_count": len(lane_lines),
            "spatial_lane_count": len(spatial_tracks),
            "cluster_count": len(clusters),
            "edge_count": int(cv2.countNonZero(masked_edges)),
            "frame_width": int(width),
            "frame_height": int(height),
            "processing_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "message": message,
        }

    def empty_result(
        self,
        frame_id: int = 0,
        frame_width: int = 0,
        frame_height: int = 0,
        status: str = "disabled",
        message: str = "正常道路模式未启用。",
    ) -> Dict[str, Any]:
        return {
            "enabled": False,
            "mode": "normal_only",
            "frame_id": int(frame_id),
            "updated_at": time.time(),
            "status": status,
            "progress": 0,
            "roi": [],
            "roi_normalized": [list(point) for point in self.roi_normalized],
            "candidate_lines": [],
            "lane_lines": [],
            "stable_lane_count": 0,
            "spatial_lane_count": 0,
            "cluster_count": 0,
            "edge_count": 0,
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "processing_ms": 0.0,
            "message": message,
        }


def point_inside_polygon(point: Tuple[float, float], polygon: Sequence[Sequence[int]]) -> bool:
    if len(polygon) < 3:
        return False
    contour = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0

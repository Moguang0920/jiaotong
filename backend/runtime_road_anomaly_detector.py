# -*- coding: utf-8 -*-
"""运行时道路异常检测器。

设计目标：
1. 只在 normal.onnx 道路异常模式运行。
2. 与 NormalLaneDetector 共用同一组归一化 ROI 点。
3. ROI 确认后，不读取测试阶段 road_anomaly_data 中的旧基准；
   而是直接从当前正在运行的视频源采集正常道路多基准。
4. normal.onnx 的车辆检测结果只用于生成车辆忽略 Mask。
5. 基准完成后，使用“每张基准直接 ORB + Homography”的准确逻辑，
   再执行 LAB/纹理/边缘差分、多基准投票和时间连续确认。

本模块没有自己的 VideoCapture。它只消费 plate_runtime_backend.py
已经读取的最新帧，因此不会重新连接手机，也不会影响其他模型切换。
"""

from __future__ import annotations

import json
import math
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from .normal_lane_detector import normalized_roi_to_pixels, sanitize_normalized_roi
except ImportError:
    from normal_lane_detector import normalized_roi_to_pixels, sanitize_normalized_roi


# ---------------------------------------------------------------------------
# 运行时采集参数
# ---------------------------------------------------------------------------

REFERENCE_TARGET = 12
REFERENCE_MIN_INTERVAL_SECONDS = 0.42
REFERENCE_MAX_VEHICLE_OCCUPANCY = 0.42
REFERENCE_MIN_SHARPNESS = 16.0
REFERENCE_RELAX_AFTER_SECONDS = 12.0

# 异常检测降频。独立线程会持续消费最新帧，但完整多基准比较不必每帧执行。
ANOMALY_MIN_INTERVAL_SECONDS = 0.24

# 多基准选择与投票
REFERENCE_SHORTLIST_COUNT = 7
TOP_REFERENCE_COUNT = 3
MIN_REFERENCE_VOTES = 2

# ORB / Homography
ORB_FEATURES = 5000
ORB_LOWE_RATIO = 0.76
MIN_GOOD_MATCHES = 22
MIN_INLIERS = 14
MIN_INLIER_RATIO = 0.25
MIN_WARP_AREA_RATIO = 0.70
MAX_WARP_AREA_RATIO = 1.35
MAX_CENTER_SHIFT_RATIO = 0.27
MAX_PERSPECTIVE_TERM = 0.005

# Mask 与差分
ROAD_BOUNDARY_ERODE = 5
VEHICLE_EXTRA_DILATE = 7
BASE_DIFF_THRESHOLD = 37.0
MAX_ADAPTIVE_THRESHOLD = 72.0
MAD_MULTIPLIER = 4.2
MAD_EXTRA = 7.0
OPEN_KERNEL_SIZE = 3
CLOSE_KERNEL_SIZE = 9

# 连通区域
BOTTOM_MIN_AREA = 170
TOP_MIN_AREA = 34
MAX_COMPONENT_ROAD_RATIO = 0.16
MIN_COMPONENT_MEAN_SCORE = 42.0

# 时间确认
CONFIRM_MIN_SECONDS = 1.25
CONFIRM_MIN_HITS = 5
MAX_TRACK_MISSES = 8
TRACK_IOU_THRESHOLD = 0.10
TRACK_CENTER_DISTANCE_RATIO = 1.25

# 只并行独立的基准任务，不改变算法语义。
ALIGNMENT_WORKERS = 4
DIFFERENCE_WORKERS = 3


@dataclass
class LiveReference:
    reference_id: int
    name: str
    image: np.ndarray
    road_mask: np.ndarray
    vehicle_mask: np.ndarray
    scene_feature: np.ndarray
    gray_feature: np.ndarray
    keypoints: Sequence[Any]
    descriptors: np.ndarray
    captured_at: float


@dataclass
class AlignedReference:
    entry: LiveReference
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
    event_registered: bool = False
    mean_score: float = 0.0
    vote_mean: float = 0.0
    matched_this_update: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)


def _imwrite_unicode(path: Path, image: np.ndarray, params: Optional[List[int]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image, params or [])
    if not ok:
        raise RuntimeError(f"图片编码失败：{path}")
    encoded.tofile(str(path))


def _polygon_to_mask(points: np.ndarray, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if points is not None and len(points) >= 3:
        cv2.fillPoly(mask, [points.astype(np.int32).reshape((-1, 1, 2))], 255)
    return mask


def _build_scene_feature(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA)
    equalized = cv2.equalizeHist(small)
    edges = cv2.Canny(equalized, 45, 130)
    histogram = cv2.calcHist([gray], [0], None, [16], [0, 256]).reshape(-1)
    histogram = histogram / max(float(histogram.sum()), 1.0)
    feature = np.concatenate([
        equalized.reshape(-1).astype(np.float32) / 255.0,
        edges.reshape(-1).astype(np.float32) / 255.0 * 0.75,
        histogram.astype(np.float32) * 4.0,
    ])
    return feature.astype(np.float32)


def _expanded_vehicle_mask(
    boxes: Sequence[Dict[str, Any]],
    road_mask: np.ndarray,
    frame_id: int,
) -> np.ndarray:
    height, width = road_mask.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for box in boxes or []:
        sem = str(box.get("semantic_type", "") or "").strip().lower()
        normal_status = str(box.get("normal_road_status", "") or "").strip().lower()
        if sem != "vehicle" and normal_status != "normal_vehicle":
            continue

        box_frame_id = int(box.get("frame_id", frame_id) or frame_id)
        if frame_id and box_frame_id and abs(frame_id - box_frame_id) > 5:
            continue

        raw = box.get("bbox") or []
        if len(raw) < 4:
            continue
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in raw[:4]]
        except Exception:
            continue

        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        # 车辆底部中心或框内一定面积位于道路中，才加入忽略区域。
        cx = max(0, min(width - 1, int(round((x1 + x2) / 2.0))))
        by = max(0, min(height - 1, y2 - 1))
        crop = road_mask[y1:y2, x1:x2]
        intersection = float(np.count_nonzero(crop)) / max(float((x2 - x1) * (y2 - y1)), 1.0)
        if road_mask[by, cx] == 0 and intersection < 0.10:
            continue

        bw = x2 - x1
        bh = y2 - y1
        ex = int(round(bw * 0.13))
        et = int(round(bh * 0.09))
        eb = int(round(bh * 0.17))
        x1 = max(0, x1 - ex)
        x2 = min(width - 1, x2 + ex)
        y1 = max(0, y1 - et)
        y2 = min(height - 1, y2 + eb)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    mask = cv2.bitwise_and(mask, road_mask)
    return mask


def _bbox_iou(first: Tuple[int, int, int, int], second: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return intersection / max(a_area + b_area - intersection, 1)


def _bbox_center_distance(
    first: Tuple[int, int, int, int],
    second: Tuple[int, int, int, int],
) -> Tuple[float, float]:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    first_center = np.asarray([(ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0], dtype=np.float32)
    second_center = np.asarray([(bx1 + bx2) / 2.0, (by1 + by2) / 2.0], dtype=np.float32)
    distance = float(np.linalg.norm(first_center - second_center))
    scale = max(
        math.hypot(ax2 - ax1, ay2 - ay1),
        math.hypot(bx2 - bx1, by2 - by1),
        1.0,
    )
    return distance, scale


class TemporalAnomalyTracker:
    def __init__(self) -> None:
        self.tracks: List[AnomalyTrack] = []
        self.next_track_id = 1
        self.total_confirmed_events = 0

    def reset(self) -> None:
        self.tracks.clear()
        self.next_track_id = 1
        self.total_confirmed_events = 0

    def update(self, candidates: List[CandidateRegion], timestamp: float) -> None:
        for track in self.tracks:
            track.matched_this_update = False

        pairs: List[Tuple[float, int, int]] = []
        for track_index, track in enumerate(self.tracks):
            for candidate_index, candidate in enumerate(candidates):
                iou = _bbox_iou(track.bbox, candidate.bbox)
                distance, scale = _bbox_center_distance(track.bbox, candidate.bbox)
                close = distance <= scale * TRACK_CENTER_DISTANCE_RATIO
                if iou >= TRACK_IOU_THRESHOLD or close:
                    pairs.append(((1.0 - iou) * 100.0 + distance, track_index, candidate_index))

        pairs.sort(key=lambda item: item[0])
        used_tracks = set()
        used_candidates = set()

        for _, track_index, candidate_index in pairs:
            if track_index in used_tracks or candidate_index in used_candidates:
                continue
            track = self.tracks[track_index]
            candidate = candidates[candidate_index]
            old_box = np.asarray(track.bbox, dtype=np.float32)
            new_box = np.asarray(candidate.bbox, dtype=np.float32)
            smoothed = 0.58 * old_box + 0.42 * new_box
            track.bbox = tuple(int(round(v)) for v in smoothed)
            track.last_seen = timestamp
            track.hits += 1
            track.misses = 0
            track.mean_score = (
                0.70 * track.mean_score + 0.30 * candidate.mean_score
                if track.hits > 1 else candidate.mean_score
            )
            track.vote_mean = (
                0.70 * track.vote_mean + 0.30 * candidate.vote_mean
                if track.hits > 1 else candidate.vote_mean
            )
            track.matched_this_update = True
            if track.hits >= CONFIRM_MIN_HITS and track.duration >= CONFIRM_MIN_SECONDS:
                track.confirmed = True
                if not track.event_registered:
                    track.event_registered = True
                    self.total_confirmed_events += 1
            used_tracks.add(track_index)
            used_candidates.add(candidate_index)

        for index, track in enumerate(self.tracks):
            if index not in used_tracks:
                track.misses += 1
                track.matched_this_update = False

        for index, candidate in enumerate(candidates):
            if index in used_candidates:
                continue
            self.tracks.append(
                AnomalyTrack(
                    track_id=self.next_track_id,
                    bbox=candidate.bbox,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    mean_score=candidate.mean_score,
                    vote_mean=candidate.vote_mean,
                )
            )
            self.next_track_id += 1

        self.tracks = [track for track in self.tracks if track.misses <= MAX_TRACK_MISSES]

    def active_tracks(self) -> List[AnomalyTrack]:
        return [track for track in self.tracks if track.misses <= MAX_TRACK_MISSES]


class RuntimeRoadAnomalyDetector:
    """有状态的实时基准采集与道路异常检测器。"""

    def __init__(self, output_root: Path) -> None:
        self.lock = threading.RLock()
        self.output_root = Path(output_root)
        self.reference_dir = self.output_root / "live_baseline"
        self.roi_normalized: List[List[float]] = []
        self.source = ""
        self.source_type = ""
        self.session_id = ""
        self.generation = 0
        self.status = "waiting_roi"
        self.message = "请先选择四点道路 ROI。"
        self.references: List[LiveReference] = []
        self.calibration_started_at = 0.0
        self.last_reference_at = 0.0
        self.last_anomaly_at = 0.0
        self.last_processed_frame_id = -1
        self.last_result: Dict[str, Any] = {}
        # 前端“道路辅助视图”右侧使用的轻量调试图。
        # 颜色逻辑沿用 multi_reference_road_anomaly_demo.py：
        # 低差异区域以蓝色为主，异常投票区域为红色，车辆忽略区为亮蓝色。
        self.debug_preview_png: bytes = b""
        self.debug_preview_frame_id = -1
        self.tracker = TemporalAnomalyTracker()
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
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.reset()

    def _normal_gray(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray)

    def reset(self, message: str = "请先选择四点道路 ROI。") -> None:
        with self.lock:
            self.generation += 1
            self.roi_normalized = []
            self.source = ""
            self.source_type = ""
            self.session_id = ""
            self.status = "waiting_roi"
            self.message = message
            self.references = []
            self.calibration_started_at = 0.0
            self.last_reference_at = 0.0
            self.last_anomaly_at = 0.0
            self.last_processed_frame_id = -1
            self.debug_preview_png = b""
            self.debug_preview_frame_id = -1
            self.tracker.reset()
            self.last_result = self.empty_result(status="waiting_roi", message=message)

    def configure(
        self,
        normalized_points: Sequence[Sequence[float]],
        source: str = "",
        source_type: str = "",
    ) -> bool:
        cleaned = sanitize_normalized_roi(normalized_points)
        if len(cleaned) != 4:
            return False

        with self.lock:
            self.generation += 1
            self.roi_normalized = [list(point) for point in cleaned]
            self.source = str(source or "")
            self.source_type = str(source_type or "")
            self.session_id = time.strftime("%Y%m%d_%H%M%S")
            self.status = "collecting"
            self.message = "四点 ROI 已确认，正在从当前视频源实时采集正常道路基准。"
            self.references = []
            self.calibration_started_at = time.time()
            self.last_reference_at = 0.0
            self.last_anomaly_at = 0.0
            self.last_processed_frame_id = -1
            self.debug_preview_png = b""
            self.debug_preview_frame_id = -1
            self.tracker.reset()

            if self.reference_dir.exists():
                shutil.rmtree(self.reference_dir, ignore_errors=True)
            self.reference_dir.mkdir(parents=True, exist_ok=True)
            self.last_result = self.empty_result(status="collecting", message=self.message)
        return True

    def rebuild(self) -> bool:
        with self.lock:
            points = [list(point) for point in self.roi_normalized]
            source = self.source
            source_type = self.source_type
        if len(points) != 4:
            return False
        return self.configure(points, source, source_type)

    @property
    def configured(self) -> bool:
        return len(self.roi_normalized) == 4

    def _update_debug_preview(
        self,
        frame_id: int,
        road_mask: np.ndarray,
        vehicle_mask: Optional[np.ndarray] = None,
        voted_mask: Optional[np.ndarray] = None,
        score: Optional[np.ndarray] = None,
    ) -> None:
        """生成与离线 multi-reference 脚本一致的蓝/红差分辅助图。"""
        if road_mask is None or road_mask.size == 0:
            return

        height, width = road_mask.shape[:2]
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (8, 14, 24)

        if score is not None and score.shape[:2] == road_mask.shape[:2]:
            normalized_score = np.clip(score, 0, 100)
            normalized_score = (normalized_score / 100.0 * 255.0).astype(np.uint8)
            heat = cv2.applyColorMap(normalized_score, cv2.COLORMAP_JET)
            panel[road_mask > 0] = heat[road_mask > 0]
        else:
            # 基准采集阶段没有差分分数，先用蓝色显示有效道路区域。
            panel[road_mask > 0] = (205, 72, 24)

        if voted_mask is not None and voted_mask.shape[:2] == road_mask.shape[:2]:
            panel[voted_mask > 0] = (0, 0, 255)

        if vehicle_mask is not None and vehicle_mask.shape[:2] == road_mask.shape[:2]:
            panel[vehicle_mask > 0] = (255, 80, 20)

        # 控制接口流量；调试图只负责辅助观察，不需要保持原始分辨率。
        max_width = 520
        if width > max_width:
            scale = max_width / float(width)
            panel = cv2.resize(
                panel,
                (max_width, max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_NEAREST,
            )

        ok, encoded = cv2.imencode(
            ".png",
            panel,
            [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
        )
        if ok:
            self.debug_preview_png = encoded.tobytes()
            self.debug_preview_frame_id = int(frame_id)

    def get_debug_preview_png(self) -> Tuple[bytes, int]:
        """返回最近一次道路差分辅助图的 PNG 数据与对应帧号。"""
        with self.lock:
            return bytes(self.debug_preview_png), int(self.debug_preview_frame_id)

    def empty_result(
        self,
        frame_id: int = 0,
        frame_width: int = 0,
        frame_height: int = 0,
        status: str = "disabled",
        message: str = "道路异常检测未启用。",
    ) -> Dict[str, Any]:
        progress = int(round(len(self.references) * 100.0 / max(REFERENCE_TARGET, 1)))
        return {
            "enabled": status not in {"disabled"},
            "status": status,
            "mode": "live_baseline_multi_reference",
            "frame_id": int(frame_id),
            "updated_at": time.time(),
            "roi_normalized": [list(point) for point in self.roi_normalized],
            "roi": [],
            "baseline_source": "current_live_stream",
            "baseline_session_id": self.session_id,
            "baseline_count": len(self.references),
            "baseline_target": REFERENCE_TARGET,
            "baseline_progress": min(100, max(0, progress)),
            "baseline_ready": self.status == "ready",
            "candidate_count": 0,
            "confirmed_count": 0,
            "total_confirmed_events": int(self.tracker.total_confirmed_events),
            "candidates": [],
            "confirmed": [],
            "tracks": [],
            "selected_references": [],
            "thresholds": [],
            "alignment_count": 0,
            "difference_pixels": 0,
            "processing_ms": 0.0,
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "message": message,
        }

    def _save_reference(self, entry: LiveReference) -> None:
        image_path = self.reference_dir / f"reference_{entry.reference_id:02d}.jpg"
        vehicle_path = self.reference_dir / f"reference_{entry.reference_id:02d}_vehicle_mask.png"
        road_path = self.reference_dir / f"reference_{entry.reference_id:02d}_road_mask.png"
        _imwrite_unicode(image_path, entry.image, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        _imwrite_unicode(vehicle_path, entry.vehicle_mask)
        _imwrite_unicode(road_path, entry.road_mask)

        metadata = {
            "session_id": self.session_id,
            "source": self.source,
            "source_type": self.source_type,
            "strategy": "live_current_source_multi_reference",
            "roi_normalized": self.roi_normalized,
            "reference_count": len(self.references),
            "reference_target": REFERENCE_TARGET,
            "references": [
                {
                    "reference_id": ref.reference_id,
                    "name": ref.name,
                    "captured_at": ref.captured_at,
                    "keypoints": len(ref.keypoints),
                }
                for ref in self.references
            ],
        }
        (self.reference_dir / "baseline.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _collect_reference(
        self,
        frame: np.ndarray,
        frame_id: int,
        road_mask: np.ndarray,
        vehicle_mask: np.ndarray,
        now: float,
    ) -> Dict[str, Any]:
        height, width = frame.shape[:2]
        elapsed = now - self.calibration_started_at
        if now - self.last_reference_at < REFERENCE_MIN_INTERVAL_SECONDS:
            return self._calibration_result(frame_id, width, height)

        road_pixels = max(int(np.count_nonzero(road_mask)), 1)
        vehicle_ratio = float(np.count_nonzero(vehicle_mask)) / float(road_pixels)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        road_values = gray[road_mask > 0]
        if road_values.size < 1000:
            self.message = "ROI 有效面积过小，请重新选择四点道路区域。"
            return self._calibration_result(frame_id, width, height)

        sharpness_map = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(np.var(sharpness_map[road_mask > 0]))

        relaxed = elapsed >= REFERENCE_RELAX_AFTER_SECONDS
        max_vehicle_ratio = 0.62 if relaxed else REFERENCE_MAX_VEHICLE_OCCUPANCY
        min_sharpness = 8.0 if relaxed else REFERENCE_MIN_SHARPNESS

        if vehicle_ratio > max_vehicle_ratio:
            self.message = (
                f"正常基准采集中：道路车辆占比 {vehicle_ratio * 100:.0f}% 较高，"
                f"等待更干净的画面（{len(self.references)}/{REFERENCE_TARGET}）。"
            )
            return self._calibration_result(frame_id, width, height)

        if sharpness < min_sharpness:
            self.message = (
                f"正常基准采集中：当前帧较模糊，等待清晰画面"
                f"（{len(self.references)}/{REFERENCE_TARGET}）。"
            )
            return self._calibration_result(frame_id, width, height)

        feature_mask = cv2.bitwise_not(
            cv2.dilate(vehicle_mask, np.ones((17, 17), np.uint8), iterations=1)
        )
        normalized_gray = self._normal_gray(frame)
        keypoints, descriptors = self.orb.detectAndCompute(normalized_gray, feature_mask)

        if descriptors is None or len(keypoints) < MIN_GOOD_MATCHES:
            self.message = (
                f"正常基准采集中：当前帧固定特征不足"
                f"（{len(self.references)}/{REFERENCE_TARGET}）。"
            )
            return self._calibration_result(frame_id, width, height)

        reference_id = len(self.references) + 1
        entry = LiveReference(
            reference_id=reference_id,
            name=f"LIVE-{reference_id:02d}",
            image=frame.copy(),
            road_mask=road_mask.copy(),
            vehicle_mask=vehicle_mask.copy(),
            scene_feature=_build_scene_feature(frame),
            gray_feature=normalized_gray,
            keypoints=keypoints,
            descriptors=descriptors,
            captured_at=now,
        )
        self.references.append(entry)
        self.last_reference_at = now
        self._save_reference(entry)

        if len(self.references) >= REFERENCE_TARGET:
            self.status = "ready"
            self.message = (
                f"实时正常道路基准采集完成：{len(self.references)} 张。"
                "车道线绘制与道路异常检测正在共用同一四点 ROI。"
            )
            self.tracker.reset()
        else:
            self.message = (
                f"正在从当前视频源采集正常道路基准："
                f"{len(self.references)}/{REFERENCE_TARGET}。"
            )

        return self._calibration_result(frame_id, width, height)

    def _calibration_result(self, frame_id: int, width: int, height: int) -> Dict[str, Any]:
        roi_pixels = normalized_roi_to_pixels(self.roi_normalized, width, height)
        result = self.empty_result(
            frame_id=frame_id,
            frame_width=width,
            frame_height=height,
            status=self.status,
            message=self.message,
        )
        result["roi"] = [[int(x), int(y)] for x, y in roi_pixels.tolist()]
        result["enabled"] = True
        result["baseline_ready"] = self.status == "ready"
        self.last_result = result
        return result

    def _validate_homography(self, homography: np.ndarray, width: int, height: int) -> bool:
        corners = np.asarray(
            [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        contour = np.round(transformed).astype(np.int32).reshape((-1, 1, 2))
        if not cv2.isContourConvex(contour):
            return False

        original_area = float(width * height)
        transformed_area = abs(float(cv2.contourArea(transformed.astype(np.float32))))
        area_ratio = transformed_area / max(original_area, 1.0)
        source_center = np.asarray([width / 2.0, height / 2.0], dtype=np.float32)
        target_center = transformed.mean(axis=0)
        center_shift = float(np.linalg.norm(target_center - source_center))
        max_center_shift = math.hypot(width, height) * MAX_CENTER_SHIFT_RATIO
        perspective_term = max(abs(float(homography[2, 0])), abs(float(homography[2, 1])))

        return bool(
            MIN_WARP_AREA_RATIO <= area_ratio <= MAX_WARP_AREA_RATIO
            and center_shift <= max_center_shift
            and perspective_term <= MAX_PERSPECTIVE_TERM
        )

    def _align_one(
        self,
        entry: LiveReference,
        current_keypoints: Sequence[Any],
        current_descriptors: np.ndarray,
        width: int,
        height: int,
    ) -> Optional[AlignedReference]:
        try:
            knn_matches = self.matcher.knnMatch(entry.descriptors, current_descriptors, k=2)
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
                [entry.keypoints[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            current_points = np.float32(
                [current_keypoints[m.trainIdx].pt for m in good_matches]
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
            if inliers < MIN_INLIERS or inlier_ratio < MIN_INLIER_RATIO:
                return None
            if not self._validate_homography(homography, width, height):
                return None

            warped_image = cv2.warpPerspective(
                entry.image, homography, (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            warped_road_mask = cv2.warpPerspective(
                entry.road_mask, homography, (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            warped_vehicle_mask = cv2.warpPerspective(
                entry.vehicle_mask, homography, (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            _, warped_road_mask = cv2.threshold(warped_road_mask, 127, 255, cv2.THRESH_BINARY)
            _, warped_vehicle_mask = cv2.threshold(warped_vehicle_mask, 127, 255, cv2.THRESH_BINARY)

            return AlignedReference(
                entry=entry,
                homography=homography,
                good_matches=len(good_matches),
                inliers=inliers,
                inlier_ratio=float(inlier_ratio),
                warped_image=warped_image,
                warped_road_mask=warped_road_mask,
                warped_vehicle_mask=warped_vehicle_mask,
            )
        except Exception:
            return None

    def _align_best(
        self,
        current_frame: np.ndarray,
        current_vehicle_mask: np.ndarray,
    ) -> List[AlignedReference]:
        height, width = current_frame.shape[:2]
        current_gray = self._normal_gray(current_frame)
        current_orb_mask = cv2.bitwise_not(
            cv2.dilate(current_vehicle_mask, np.ones((17, 17), np.uint8), iterations=1)
        )
        current_keypoints, current_descriptors = self.orb.detectAndCompute(
            current_gray, current_orb_mask
        )
        if current_descriptors is None or len(current_keypoints) < MIN_GOOD_MATCHES:
            return []

        current_feature = _build_scene_feature(current_frame)
        scored = sorted(
            (
                (float(np.mean(np.abs(current_feature - entry.scene_feature))), entry)
                for entry in self.references
            ),
            key=lambda item: item[0],
        )
        shortlist = [entry for _, entry in scored[:min(REFERENCE_SHORTLIST_COUNT, len(scored))]]

        results: List[AlignedReference] = []
        with ThreadPoolExecutor(max_workers=min(ALIGNMENT_WORKERS, len(shortlist))) as executor:
            futures = [
                executor.submit(
                    self._align_one,
                    entry,
                    current_keypoints,
                    current_descriptors,
                    width,
                    height,
                )
                for entry in shortlist
            ]
            for future in as_completed(futures):
                aligned = future.result()
                if aligned is not None:
                    results.append(aligned)

        results.sort(key=lambda item: (item.inliers, item.inlier_ratio), reverse=True)
        return results[:min(TOP_REFERENCE_COUNT, len(results))]

    @staticmethod
    def _create_effective_mask(
        current_road_mask: np.ndarray,
        current_vehicle_mask: np.ndarray,
        aligned: AlignedReference,
    ) -> np.ndarray:
        reference_vehicle = aligned.warped_vehicle_mask
        current_vehicle = current_vehicle_mask
        if VEHICLE_EXTRA_DILATE > 0:
            kernel = np.ones((VEHICLE_EXTRA_DILATE, VEHICLE_EXTRA_DILATE), dtype=np.uint8)
            current_vehicle = cv2.dilate(current_vehicle, kernel, iterations=1)
            reference_vehicle = cv2.dilate(reference_vehicle, kernel, iterations=1)

        valid = cv2.bitwise_and(current_road_mask, aligned.warped_road_mask)
        invalid_vehicles = cv2.bitwise_or(current_vehicle, reference_vehicle)
        valid = cv2.bitwise_and(valid, cv2.bitwise_not(invalid_vehicles))

        if ROAD_BOUNDARY_ERODE > 0:
            kernel = np.ones((ROAD_BOUNDARY_ERODE, ROAD_BOUNDARY_ERODE), dtype=np.uint8)
            valid = cv2.erode(valid, kernel, iterations=1)
        return valid

    @staticmethod
    def _normalize_luminance(
        current_l: np.ndarray,
        reference_l: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        valid = valid_mask > 0
        if np.count_nonzero(valid) < 100:
            return current_l, reference_l

        current_values = current_l[valid].astype(np.float32)
        reference_values = reference_l[valid].astype(np.float32)
        current_mean = float(current_values.mean())
        reference_mean = float(reference_values.mean())
        current_std = max(float(current_values.std()), 6.0)
        reference_std = max(float(reference_values.std()), 6.0)

        adjusted_reference = (
            (reference_l.astype(np.float32) - reference_mean)
            * (current_std / reference_std)
            + current_mean
        )
        adjusted_reference = np.clip(adjusted_reference, 0, 255).astype(np.uint8)
        return current_l, adjusted_reference

    def _compute_difference_score(
        self,
        current_frame: np.ndarray,
        reference_frame: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        current_lab = cv2.cvtColor(current_frame, cv2.COLOR_BGR2LAB)
        reference_lab = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2LAB)
        current_l, current_a, current_b = cv2.split(current_lab)
        reference_l, reference_a, reference_b = cv2.split(reference_lab)

        current_l = self.clahe.apply(current_l)
        reference_l = self.clahe.apply(reference_l)
        current_l, reference_l = self._normalize_luminance(
            current_l, reference_l, valid_mask
        )

        luminance_diff = cv2.absdiff(current_l, reference_l).astype(np.float32)
        color_a_diff = cv2.absdiff(current_a, reference_a).astype(np.float32)
        color_b_diff = cv2.absdiff(current_b, reference_b).astype(np.float32)
        color_diff = (color_a_diff + color_b_diff) * 0.5

        current_detail = cv2.absdiff(
            current_l, cv2.GaussianBlur(current_l, (0, 0), 5.0)
        ).astype(np.float32)
        reference_detail = cv2.absdiff(
            reference_l, cv2.GaussianBlur(reference_l, (0, 0), 5.0)
        ).astype(np.float32)
        texture_diff = cv2.absdiff(
            current_detail.astype(np.uint8), reference_detail.astype(np.uint8)
        ).astype(np.float32)

        current_edges = cv2.Canny(current_l, 60, 145)
        reference_edges = cv2.Canny(reference_l, 60, 145)
        edge_diff = cv2.absdiff(current_edges, reference_edges).astype(np.float32)

        score = (
            0.50 * luminance_diff
            + 0.20 * color_diff
            + 0.18 * texture_diff
            + 0.12 * edge_diff
        )
        score = cv2.medianBlur(
            np.clip(score, 0, 255).astype(np.uint8), 5
        ).astype(np.float32)
        score[valid_mask == 0] = 0.0

        valid_values = score[valid_mask > 0]
        if valid_values.size == 0:
            return score, BASE_DIFF_THRESHOLD

        median = float(np.median(valid_values))
        mad = float(np.median(np.abs(valid_values - median)))
        adaptive_threshold = max(
            BASE_DIFF_THRESHOLD,
            median + MAD_MULTIPLIER * mad + MAD_EXTRA,
        )
        adaptive_threshold = min(adaptive_threshold, MAX_ADAPTIVE_THRESHOLD)
        return score, float(adaptive_threshold)

    def _compare_one(
        self,
        current_frame: np.ndarray,
        current_road_mask: np.ndarray,
        current_vehicle_mask: np.ndarray,
        aligned: AlignedReference,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        valid_mask = self._create_effective_mask(
            current_road_mask, current_vehicle_mask, aligned
        )
        score, threshold = self._compute_difference_score(
            current_frame, aligned.warped_image, valid_mask
        )
        binary = ((score >= threshold) & (valid_mask > 0)).astype(np.uint8) * 255
        return binary, score, float(threshold), valid_mask

    @staticmethod
    def _perspective_min_area(center_y: float, image_height: int) -> int:
        ratio = np.clip(center_y / max(image_height - 1, 1), 0.0, 1.0)
        return int(round(TOP_MIN_AREA + (BOTTOM_MIN_AREA - TOP_MIN_AREA) * ratio * ratio))

    def _extract_candidates(
        self,
        binary_mask: np.ndarray,
        combined_score: np.ndarray,
        vote_map: np.ndarray,
        road_mask: np.ndarray,
    ) -> List[CandidateRegion]:
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        road_pixels = max(int(np.count_nonzero(road_mask)), 1)
        candidates: List[CandidateRegion] = []

        for label_id in range(1, count):
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            width = int(stats[label_id, cv2.CC_STAT_WIDTH])
            height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            center_y = float(centroids[label_id][1])

            if area < self._perspective_min_area(center_y, binary_mask.shape[0]):
                continue
            if area / float(road_pixels) > MAX_COMPONENT_ROAD_RATIO:
                continue
            if width < 6 or height < 6:
                continue

            component = labels == label_id
            mean_score = float(combined_score[component].mean())
            vote_mean = float(vote_map[component].mean())
            if mean_score < MIN_COMPONENT_MEAN_SCORE:
                continue

            # 细长车道线残差保护：只有分数非常高时才允许极细长区域进入。
            aspect = max(width / max(height, 1), height / max(width, 1))
            if aspect > 9.0 and min(width, height) < 14 and mean_score < 62.0:
                continue

            candidates.append(
                CandidateRegion(
                    bbox=(x, y, x + width, y + height),
                    area=area,
                    mean_score=mean_score,
                    vote_mean=vote_mean,
                )
            )
        return candidates

    def _run_difference(
        self,
        current_frame: np.ndarray,
        current_road_mask: np.ndarray,
        current_vehicle_mask: np.ndarray,
        aligned_references: List[AlignedReference],
    ) -> Tuple[np.ndarray, np.ndarray, List[CandidateRegion], List[float], np.ndarray]:
        if len(aligned_references) < MIN_REFERENCE_VOTES:
            empty = np.zeros_like(current_road_mask)
            return empty, empty.astype(np.float32), [], [], empty.astype(np.float32)

        with ThreadPoolExecutor(
            max_workers=min(DIFFERENCE_WORKERS, len(aligned_references))
        ) as executor:
            results = list(
                executor.map(
                    lambda aligned: self._compare_one(
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
        thresholds = []
        valid_union = np.zeros_like(current_road_mask)

        for binary, score, threshold, valid_mask in results:
            valid_union = cv2.bitwise_or(valid_union, valid_mask)
            vote_layers.append(binary > 0)
            score_layers.append(score)
            thresholds.append(float(threshold))

        vote_map = np.sum(np.stack(vote_layers, axis=0), axis=0).astype(np.uint8)
        combined_score = np.mean(np.stack(score_layers, axis=0), axis=0).astype(np.float32)
        voted_binary = (
            (vote_map >= MIN_REFERENCE_VOTES) & (valid_union > 0)
        ).astype(np.uint8) * 255

        open_kernel = np.ones((OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE), dtype=np.uint8)
        close_kernel = np.ones((CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE), dtype=np.uint8)
        voted_binary = cv2.morphologyEx(
            voted_binary, cv2.MORPH_OPEN, open_kernel, iterations=1
        )
        voted_binary = cv2.morphologyEx(
            voted_binary, cv2.MORPH_CLOSE, close_kernel, iterations=1
        )

        candidates = self._extract_candidates(
            voted_binary,
            combined_score,
            vote_map.astype(np.float32),
            current_road_mask,
        )
        return voted_binary, combined_score, candidates, thresholds, vote_map.astype(np.float32)

    def _track_to_dict(self, track: AnomalyTrack) -> Dict[str, Any]:
        return {
            "track_id": int(track.track_id),
            "event_id": f"ROAD-{track.track_id:03d}",
            "bbox": [int(v) for v in track.bbox],
            "confirmed": bool(track.confirmed),
            "duration_s": round(float(track.duration), 2),
            "hits": int(track.hits),
            "misses": int(track.misses),
            "score": round(float(track.mean_score), 2),
            "vote_mean": round(float(track.vote_mean), 2),
            "label": "道路异常" if track.confirmed else "异常候选",
        }

    def process(
        self,
        frame: np.ndarray,
        frame_id: int,
        vehicle_boxes: Sequence[Dict[str, Any]],
        frame_ts: float = 0.0,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("道路异常检测收到空帧")

        with self.lock:
            height, width = frame.shape[:2]
            if not self.configured:
                result = self.empty_result(
                    frame_id=frame_id,
                    frame_width=width,
                    frame_height=height,
                    status="waiting_roi",
                    message="请先在当前视频帧上选择四个道路 ROI 顶点。",
                )
                self.last_result = result
                return result

            roi_pixels = normalized_roi_to_pixels(self.roi_normalized, width, height)
            road_mask = _polygon_to_mask(roi_pixels, width, height)
            vehicle_mask = _expanded_vehicle_mask(vehicle_boxes, road_mask, frame_id)
            now = time.time()

            if self.status == "collecting":
                self._update_debug_preview(
                    frame_id=frame_id,
                    road_mask=road_mask,
                    vehicle_mask=vehicle_mask,
                )
                result = self._collect_reference(
                    frame, frame_id, road_mask, vehicle_mask, now
                )
                result["processing_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
                self.last_result = result
                return result

            if self.status != "ready" or len(self.references) < MIN_REFERENCE_VOTES:
                result = self._calibration_result(frame_id, width, height)
                result["processing_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
                self.last_result = result
                return result

            if (
                frame_id == self.last_processed_frame_id
                or now - self.last_anomaly_at < ANOMALY_MIN_INTERVAL_SECONDS
            ):
                cached = dict(self.last_result)
                cached["frame_id"] = int(frame_id)
                cached["updated_at"] = now
                cached["roi"] = [[int(x), int(y)] for x, y in roi_pixels.tolist()]
                return cached

            self.last_processed_frame_id = int(frame_id)
            self.last_anomaly_at = now

            aligned = self._align_best(frame, vehicle_mask)
            voted_mask, combined_score, candidates, thresholds, vote_map = self._run_difference(
                frame, road_mask, vehicle_mask, aligned
            )
            self._update_debug_preview(
                frame_id=frame_id,
                road_mask=road_mask,
                vehicle_mask=vehicle_mask,
                voted_mask=voted_mask,
                score=combined_score,
            )
            timestamp = float(frame_ts or now)
            self.tracker.update(candidates, timestamp)

            active_tracks = self.tracker.active_tracks()
            candidate_tracks = [track for track in active_tracks if not track.confirmed]
            confirmed_tracks = [track for track in active_tracks if track.confirmed]

            if len(aligned) < MIN_REFERENCE_VOTES:
                message = (
                    f"实时基准已就绪，但当前帧只有 {len(aligned)} 张基准成功配准，"
                    "正在等待更稳定的画面。"
                )
            elif confirmed_tracks:
                message = (
                    f"道路异常检测：确认 {len(confirmed_tracks)} 处异常，"
                    f"另有 {len(candidate_tracks)} 处候选；"
                    f"车道线与异常检测共用同一四点 ROI。"
                )
            else:
                message = (
                    f"道路异常检测运行中：{len(candidate_tracks)} 处短时候选，"
                    "未确认持续异常。"
                )

            tracks_dict = [self._track_to_dict(track) for track in active_tracks]
            result = {
                "enabled": True,
                "status": "ready",
                "mode": "live_baseline_multi_reference",
                "frame_id": int(frame_id),
                "updated_at": now,
                "roi_normalized": [list(point) for point in self.roi_normalized],
                "roi": [[int(x), int(y)] for x, y in roi_pixels.tolist()],
                "baseline_source": "current_live_stream",
                "baseline_session_id": self.session_id,
                "baseline_count": len(self.references),
                "baseline_target": REFERENCE_TARGET,
                "baseline_progress": 100,
                "baseline_ready": True,
                "candidate_count": len(candidate_tracks),
                "confirmed_count": len(confirmed_tracks),
                "total_confirmed_events": int(self.tracker.total_confirmed_events),
                "candidates": [
                    item for item in tracks_dict if not bool(item.get("confirmed"))
                ],
                "confirmed": [
                    item for item in tracks_dict if bool(item.get("confirmed"))
                ],
                "tracks": tracks_dict,
                "selected_references": [aligned_ref.entry.name for aligned_ref in aligned],
                "thresholds": [round(float(value), 2) for value in thresholds],
                "alignment_count": len(aligned),
                "difference_pixels": int(np.count_nonzero(voted_mask)),
                "processing_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "frame_width": int(width),
                "frame_height": int(height),
                "message": message,
            }
            self.message = message
            self.last_result = result
            return result

# -*- coding: utf-8 -*-
"""
智慧交通视觉感知系统 - ONNX 并行版车牌检测后端

核心改进：
1. 车牌检测优先加载 best(1).onnx，不再使用 best(1).pt 串行推理。
2. 三线程解耦：视频读取线程、YOLO ONNX 推理线程、PaddleOCR 识别线程。
3. 前端 MJPEG 显示不等待 AI 推理，始终展示最新视频帧 + 当前 YOLO 实时框。
4. OCR 不再控制画框，只作为 track_id 的文字缓存；当前帧 YOLO 框立即显示，OCR 结果异步复用。
5. OCR 不再每帧执行；未稳定车牌前 6 次最高约 4 次/秒，之后自动降频，稳定后每 2 秒复核。
5. OCR 默认使用 PP-OCRv6_tiny_rec，失败时自动回退默认 TextRecognition。

模型放置：
- 推荐：项目根目录/models/best(1).onnx
- 也兼容项目根目录下的同名模型，或通过环境变量指定模型路径

启动：
- 由 Electron main.js 自动启动，或手动执行：
  python backend/plate_runtime_backend.py
"""

from __future__ import annotations

import ast
import base64
import json
import math
import os
import site
import sqlite3
import hashlib
import uuid
import queue
import re
import sys
import time
import threading
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

try:
    from .normal_lane_detector import (
        NormalLaneDetector,
        normalized_roi_to_pixels,
        point_inside_polygon,
        sanitize_normalized_roi,
    )
except ImportError:  # 直接运行 backend/plate_runtime_backend.py 时使用同目录导入
    from normal_lane_detector import (
        NormalLaneDetector,
        normalized_roi_to_pixels,
        point_inside_polygon,
        sanitize_normalized_roi,
    )

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None


ROOT_DIR = Path(__file__).resolve().parents[1]
PARKING_EVENT_LOG_PATH = ROOT_DIR / 'runtime_data' / 'parking_events.jsonl'
DB_PATH = ROOT_DIR / "runtime_data" / "trafficvision.db"

# Windows 下 onnxruntime-gpu[cuda,cudnn] 会把 CUDA/cuDNN DLL 放在 site-packages/nvidia/**/bin 中。
# Electron 启动 Python 时，这些目录通常不在 PATH 里；如果不先 preload，Session 会报 cudnn64_9.dll missing，
# 然后静默回退 CPU。这里保留 add_dll_directory 的句柄，防止目录句柄被 GC。
_DLL_DIR_HANDLES: List[Any] = []
_CUDA_DLL_PRELOADED = False

def preload_cuda_runtime_for_onnx() -> Dict[str, Any]:
    global _CUDA_DLL_PRELOADED
    info: Dict[str, Any] = {"ok": False, "message": "", "dll_dirs": []}
    if ort is None:
        info["message"] = "onnxruntime 未导入"
        return info

    dll_dirs: List[str] = []
    try:
        for sp in site.getsitepackages():
            nvidia_root = Path(sp) / "nvidia"
            if not nvidia_root.exists():
                continue
            for sub in nvidia_root.glob("*"):
                bin_dir = sub / "bin"
                if bin_dir.exists() and any(bin_dir.glob("*.dll")):
                    dll_dirs.append(str(bin_dir))
        # 去重但保序
        dll_dirs = list(dict.fromkeys(dll_dirs))
        if os.name == "nt":
            existing_path = os.environ.get("PATH", "")
            for d in dll_dirs:
                if d not in existing_path:
                    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                try:
                    _DLL_DIR_HANDLES.append(os.add_dll_directory(d))
                except Exception:
                    pass

        if hasattr(ort, "preload_dlls"):
            # directory="" 会优先从 site-packages/nvidia 中加载 CUDA/cuDNN DLL。
            ort.preload_dlls(directory="")
        _CUDA_DLL_PRELOADED = True
        info["ok"] = True
        info["message"] = f"CUDA/cuDNN DLL preload finished, dll_dirs={len(dll_dirs)}"
        info["dll_dirs"] = dll_dirs
        return info
    except Exception as e:
        info["message"] = f"CUDA/cuDNN DLL preload failed: {repr(e)}"
        info["dll_dirs"] = dll_dirs
        return info


def _first_existing(paths: List[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]

DEFAULT_ONNX_PATH = Path(os.environ.get(
    "PLATE_ONNX_MODEL",
    _first_existing([
        ROOT_DIR / "best(1).onnx",
        ROOT_DIR / "best.onnx",
        ROOT_DIR / "models" / "best(1).onnx",
        ROOT_DIR / "models" / "best.onnx",
    ])
))

# 车辆识别/热力图前置验证模型。你当前说的文件名是 hearmap.onnx，这里同时兼容 heatmap.onnx。
DEFAULT_VEHICLE_ONNX_PATH = Path(os.environ.get(
    "VEHICLE_ONNX_MODEL",
    _first_existing([
        ROOT_DIR / "hearmap.onnx",
        ROOT_DIR / "heatmap.onnx",
        ROOT_DIR / "models" / "hearmap.onnx",
        ROOT_DIR / "models" / "heatmap.onnx",
    ])
))

# 禁停区域/禁止停车区域检测模型。第三阶段先只做模型加载与检测框验证，后续再把车辆轨迹与禁停区域做规则联动。
DEFAULT_STOP_ONNX_PATH = Path(os.environ.get(
    "STOP_ONNX_MODEL",
    _first_existing([
        ROOT_DIR / "stop.onnx",
        ROOT_DIR / "models" / "stop.onnx",
    ])
))

# 普通/正常区域检测模型。第四阶段先只做模型加载与检测框验证，后续可和 stop.onnx / 车辆轨迹做规则联动。
DEFAULT_NORMAL_ONNX_PATH = Path(os.environ.get(
    "NORMAL_ONNX_MODEL",
    _first_existing([
        ROOT_DIR / "normal.onnx",
        ROOT_DIR / "models" / "normal.onnx",
    ])
))

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "plate": {
        "name": "车牌识别模型",
        "file": DEFAULT_ONNX_PATH,
        "label": "plate",
        "display_label": "车牌",
        "uses_ocr": True,
    },
    "vehicle": {
        "name": "车辆识别模型",
        "file": DEFAULT_VEHICLE_ONNX_PATH,
        "label": "vehicle",
        "display_label": "车辆",
        "uses_ocr": False,
    },
    "stop": {
        "name": "禁停区域检测模型",
        "file": DEFAULT_STOP_ONNX_PATH,
        "label": "stop_zone",
        "display_label": "禁停区域",
        "uses_ocr": False,
    },
    "normal": {
        "name": "正常区域检测模型",
        "file": DEFAULT_NORMAL_ONNX_PATH,
        "label": "normal_zone",
        "display_label": "正常区域",
        "uses_ocr": False,
    },
}


def normalize_detector_model(value: Any) -> str:
    key = str(value or "plate").strip().lower()
    aliases = {
        "car": "vehicle",
        "cars": "vehicle",
        "heatmap": "vehicle",
        "hearmap": "vehicle",
        "vehicle_detection": "vehicle",
        "no_parking": "stop",
        "parking_forbidden": "stop",
        "forbidden_stop": "stop",
        "stop_area": "stop",
        "stop_zone": "stop",
        "stop_detection": "stop",
        "normal_area": "normal",
        "normal_zone": "normal",
        "normal_detection": "normal",
        "normal_road": "normal",
        "normal": "normal",
        "license_plate": "plate",
        "plate_detection": "plate",
    }
    key = aliases.get(key, key)
    return key if key in MODEL_REGISTRY else "plate"


def active_model_info(model_key: Optional[str] = None) -> Dict[str, Any]:
    key = normalize_detector_model(model_key if model_key is not None else getattr(STATE.config, "detector_model", "plate")) if "STATE" in globals() else normalize_detector_model(model_key)
    info = dict(MODEL_REGISTRY[key])
    info["key"] = key
    info["file"] = Path(info["file"])
    return info

DEFAULT_CAMERA_URL = os.environ.get("IP_WEBCAM_URL", "http://100.70.11.30:8080/video")
BACKEND_PORT = int(os.environ.get("TRAFFIC_BACKEND_PORT", "8765"))

PROVINCES = set("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼")
SPECIAL_PLATE_PREFIXES = {"使", "领", "警", "学", "港", "澳"}
VALID_PREFIXES = PROVINCES | SPECIAL_PLATE_PREFIXES


class StartVideoRequest(BaseModel):
    path: str
    # 仅 detector_model=normal 时使用。坐标为 0~1 归一化多边形点。
    normal_roi: List[List[float]] = []


class StartCameraRequest(BaseModel):
    url: str = DEFAULT_CAMERA_URL
    # 仅 detector_model=normal 时使用。坐标为 0~1 归一化多边形点。
    normal_roi: List[List[float]] = []


class NormalRoiPreviewRequest(BaseModel):
    source: str
    source_type: str = "file"


class NormalRoiConfigureRequest(BaseModel):
    # 0~1 归一化多边形顶点。运行中配置时不会重新打开或重连视频源。
    points: List[List[float]] = []


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "operator"
    display_name: str = ""

class LoginRequest(BaseModel):
    username: str
    password: str

class DeviceRequest(BaseModel):
    name: str
    type: str = "camera"
    stream_url: str = ""
    position: str = ""
    status: str = "offline"
    note: str = ""

class WhitelistRequest(BaseModel):
    plate_no: str
    owner: str = ""
    allow: bool = True
    note: str = ""

class ConfigItemRequest(BaseModel):
    key: str
    value: str
    description: str = ""


class BackendConfig(BaseModel):
    # 当前检测模型：plate=best(1).onnx 车牌检测；vehicle=hearmap.onnx/heatmap.onnx 车辆检测；stop=stop.onnx 禁停区域检测；normal=normal.onnx 正常区域检测。
    detector_model: str = "plate"

    # ONNX 推理参数
    conf: float = 0.35
    imgsz: int = 416
    nms_iou: float = 0.45
    yolo_interval: float = 0.12  # 每 0.12 秒最多跑一次 YOLO，约 8 FPS

    # OCR 参数：新 track 前 6 次采用 0.25 秒突发识别；仍未稳定时降到 0.60 秒；稳定后每 2 秒复核。
    ocr_min_interval: float = 0.25
    ocr_burst_max_attempts: int = 6
    ocr_unstable_interval: float = 0.60
    stable_recheck_interval: float = 2.0
    min_stable_votes: int = 3
    max_vote_history: int = 12
    min_ocr_conf: float = 0.45
    ocr_model_name: str = "PP-OCRv6_tiny_rec"

    # 显示与时序同步参数
    display_fps: float = 20.0
    jpeg_quality: int = 78
    max_ocr_queue: int = 16

    # 防止“最新视频帧 + 旧检测框/OCR”错位。
    # camera 模式默认按手持/轻微晃动场景处理，track 与框显示 TTL 更短。
    camera_track_ttl: float = 0.90
    file_track_ttl: float = 2.00
    camera_display_ttl: float = 0.35
    file_display_ttl: float = 1.20
    camera_max_frame_lag: int = 4
    file_max_frame_lag: int = 25
    camera_ocr_result_max_age: float = 0.85
    file_ocr_result_max_age: float = 3.0

    # 摄像头大幅晃动/切换位置时，旧框和旧 OCR 直接清理。
    motion_reset_enabled: bool = True
    motion_reset_score: float = 22.0
    motion_reset_min_interval: float = 0.75

    # 性能监测 / GPU Provider 诊断参数：本版核心目标是看清楚慢在哪里，而不是盲目降画质。
    # 默认优先 CUDA。TensorRT 需要额外 TensorRT DLL/版本匹配，默认关闭，避免初始化失败后误回退 CPU。
    enable_tensorrt: bool = False
    trt_fp16_enable: bool = True
    trt_engine_cache_enable: bool = True
    trt_engine_cache_path: str = "cache/trt_engine"
    warmup_runs: int = 5
    perf_monitor_enabled: bool = True
    perf_sample_window: int = 180
    perf_log_interval: float = 5.0

    # 显示链路优化：默认关闭后端复杂画框，视频流只做原始帧编码；
    # 检测框/文字通过 /api/latest 给前端 Canvas 叠加，从而砍掉 Python/PIL overlay 大头。
    server_overlay: bool = False
    overlay_debug_text: bool = False

    # 禁停检测：stop.onnx 同时输出 car / carNumber / noParking 时，
    # 只跟踪 car 与 noParking 的空间关系。车辆在禁停区连续停留达到阈值后触发告警。
    parking_violation_seconds: float = 3.0
    parking_exit_grace: float = 0.9
    parking_track_ttl: float = 5.0
    parking_zone_memory_ttl: float = 2.5
    parking_intersection_ratio: float = 0.12
    # 禁停计时必须建立在“车辆已经停下”的基础上：
    # 这里用车辆底部中心点的归一化速度判断是否静止，避免车辆只是路过禁停区也被计时。
    parking_stationary_speed_norm: float = 0.018  # 每秒移动距离 / 画面对角线，小于该值认为基本停止
    parking_stationary_confirm_seconds: float = 0.45  # 连续低速一小段时间后才认为真的停下，过滤 YOLO 抖动
    # stop.onnx 偶发只检出 carNumber 而漏掉 car 时，允许用“车牌框”作为车辆存在的代理证据。
    # 这样侧边栏仍能显示禁停跟踪状态；真正的车辆框回来后会自动优先使用 car。
    parking_plate_proxy_enabled: bool = True
    parking_plate_proxy_overlap_skip: float = 0.18


@dataclass
class PlateCandidate:
    raw: str
    cleaned: str
    confidence: float
    timestamp: float
    quality: float
    province: str = ""
    body: str = ""
    source_frame_id: int = 0
    source_frame_ts: float = 0.0


@dataclass
class PlateTrack:
    track_id: int
    bbox: Tuple[int, int, int, int]
    det_conf: float
    last_seen: float
    last_frame_id: int = 0
    last_frame_ts: float = 0.0
    created_at: float = field(default_factory=time.time)
    candidates: List[PlateCandidate] = field(default_factory=list)
    stable_text: str = ""
    stable_score: float = 0.0
    stable_at: float = 0.0
    last_ocr_time: float = 0.0
    ocr_pending: bool = False
    ocr_attempts: int = 0
    detector_model: str = "plate"
    semantic_type: str = "plate_number"
    class_id: int = 0

    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def is_recent(self, now: float, ttl: float = 2.0) -> bool:
        return now - self.last_seen <= ttl


class PlateVoting:
    """面向当前项目演示规则的多帧车牌投票。

    规则：
    1. OCR 候选只接受 6 位主体号：第 1 位必须是英文字母，后 5 位为字母或数字；
    2. OCR 可以返回“京A5678T”或“A5678T”，但省份字不参与投票；
    3. 候选阶段只展示主体号，达到稳定票数后才统一补成“京A5678T”；
    4. 除完整字符串投票外，再做逐字符位置投票，降低单个字符偶发误识别的影响。
    """

    DEMO_PROVINCE = "京"
    BODY_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{5}$")

    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""
        text = str(text).upper().replace("·", "").replace(".", "").replace(" ", "")
        return re.sub(r"[^0-9A-Z\u4e00-\u9fa5]", "", text)

    @classmethod
    def split_plate(cls, text: str) -> Tuple[str, str, float]:
        cleaned = cls.clean(text)
        if not cleaned:
            return "", "", 0.0

        province = ""
        body = ""

        # 严格长度检查：只接受 6 位主体，或 1 个中文前缀 + 6 位主体。
        if len(cleaned) == 6:
            body = cleaned
        elif len(cleaned) == 7 and "\u4e00" <= cleaned[0] <= "\u9fff":
            province = cleaned[0]
            body = cleaned[1:]
        else:
            return "", "", 0.0

        if not cls.BODY_PATTERN.fullmatch(body):
            return "", "", 0.0

        # 京字识别正确时略微加权；其他中文前缀只作为“主体有效”的证据，不参与最终省份决策。
        quality = 1.0 if province == cls.DEMO_PROVINCE else (0.96 if province else 0.93)
        return province, body, quality

    @classmethod
    def make_candidate(
        cls,
        raw_text: str,
        confidence: float,
        source_frame_id: int = 0,
        source_frame_ts: float = 0.0,
    ) -> Optional[PlateCandidate]:
        province, body, quality = cls.split_plate(raw_text)
        if not body:
            return None
        return PlateCandidate(
            raw=raw_text,
            # 候选阶段仅保存并展示 6 位主体，京字要等投票稳定后再补。
            cleaned=body,
            confidence=float(confidence or 0.0),
            timestamp=time.time(),
            quality=float(quality),
            province=province,
            body=body,
            source_frame_id=source_frame_id,
            source_frame_ts=source_frame_ts,
        )

    @classmethod
    def _aggregate(cls, track: PlateTrack) -> Dict[str, Any]:
        now = time.time()
        valid = [
            c for c in track.candidates
            if now - c.timestamp <= 20.0 and cls.BODY_PATTERN.fullmatch(str(c.body or ""))
        ]
        if not valid:
            return {
                "valid": [],
                "votes": 0,
                "best_exact": "",
                "best_exact_count": 0,
                "best_exact_share": 0.0,
                "consensus_body": "",
                "position_support": 0.0,
                "position_counts": [],
            }

        exact_scores: Dict[str, float] = {}
        exact_counts: Dict[str, int] = {}
        position_scores: List[Dict[str, float]] = [dict() for _ in range(6)]
        position_counts: List[Dict[str, int]] = [dict() for _ in range(6)]
        total_weight = 0.0

        for candidate in valid:
            body = candidate.body
            weight = max(0.01, float(candidate.confidence)) * max(0.1, float(candidate.quality))
            total_weight += weight
            exact_scores[body] = exact_scores.get(body, 0.0) + weight
            exact_counts[body] = exact_counts.get(body, 0) + 1
            for index, char in enumerate(body):
                position_scores[index][char] = position_scores[index].get(char, 0.0) + weight
                position_counts[index][char] = position_counts[index].get(char, 0) + 1

        best_exact = max(exact_scores, key=exact_scores.get) if exact_scores else ""
        best_exact_score = exact_scores.get(best_exact, 0.0)
        best_exact_count = exact_counts.get(best_exact, 0)
        best_exact_share = best_exact_score / max(total_weight, 1e-6)

        consensus_chars: List[str] = []
        support_parts: List[float] = []
        top_counts: List[int] = []
        for index in range(6):
            scores = position_scores[index]
            if not scores:
                consensus_chars.append("")
                support_parts.append(0.0)
                top_counts.append(0)
                continue
            best_char = max(scores, key=scores.get)
            consensus_chars.append(best_char)
            support_parts.append(scores[best_char] / max(total_weight, 1e-6))
            top_counts.append(position_counts[index].get(best_char, 0))

        consensus_body = "".join(consensus_chars)
        if not cls.BODY_PATTERN.fullmatch(consensus_body):
            consensus_body = ""
        position_support = sum(support_parts) / 6.0 if support_parts else 0.0

        return {
            "valid": valid,
            "votes": len(valid),
            "best_exact": best_exact,
            "best_exact_count": best_exact_count,
            "best_exact_share": best_exact_share,
            "consensus_body": consensus_body,
            "position_support": position_support,
            "position_top_counts": top_counts,
            "exact_counts": exact_counts,
            "position_counts": position_counts,
        }

    @classmethod
    def preview(cls, track: PlateTrack) -> Tuple[str, float]:
        """返回当前最高票主体号；未稳定时不补“京”。"""
        agg = cls._aggregate(track)
        if not agg["votes"]:
            return "", 0.0
        consensus = str(agg.get("consensus_body") or "")
        exact = str(agg.get("best_exact") or "")
        position_support = float(agg.get("position_support", 0.0) or 0.0)
        exact_share = float(agg.get("best_exact_share", 0.0) or 0.0)
        if consensus and position_support >= exact_share:
            return consensus, min(0.999, position_support)
        return exact, min(0.999, exact_share)

    @classmethod
    def vote(cls, track: PlateTrack, min_votes: int = 3) -> Tuple[str, float, Dict[str, Any]]:
        agg = cls._aggregate(track)
        votes = int(agg.get("votes", 0) or 0)
        debug = {k: v for k, v in agg.items() if k != "valid"}

        # 投票没有达到规定次数时，只允许前端显示 6 位主体候选，不显示京字。
        if votes < max(1, int(min_votes)):
            return "", 0.0, debug

        best_exact = str(agg.get("best_exact") or "")
        consensus = str(agg.get("consensus_body") or "")
        exact_count = int(agg.get("best_exact_count", 0) or 0)
        exact_share = float(agg.get("best_exact_share", 0.0) or 0.0)
        position_support = float(agg.get("position_support", 0.0) or 0.0)
        top_counts = [int(v or 0) for v in agg.get("position_top_counts", [])]

        selected_body = ""
        selected_score = 0.0

        # 完整字符串重复达到票数，或加权占比明显领先时，优先采用完整票。
        if best_exact and (exact_count >= min_votes or exact_share >= 0.58):
            selected_body = best_exact
            selected_score = 0.58 * exact_share + 0.42 * min(1.0, exact_count / max(1, votes))
        else:
            # 完整字符串不完全一致时，允许逐字符多数票修复一两个偶发错字。
            majority_needed = max(2, (votes + 1) // 2)
            enough_position_votes = bool(top_counts) and all(count >= majority_needed for count in top_counts)
            if consensus and enough_position_votes and position_support >= 0.62:
                selected_body = consensus
                selected_score = 0.72 * position_support + 0.28 * exact_share

        if not selected_body or not cls.BODY_PATTERN.fullmatch(selected_body):
            return "", 0.0, debug

        # 当前项目规定北京车牌：主体号稳定后才补京字。
        return f"{cls.DEMO_PROVINCE}{selected_body}", min(0.999, max(0.0, selected_score)), debug


class RuntimeState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.config = BackendConfig()

        self.status: Dict[str, Any] = {
            "backend": "starting",
            "models_ready": False,
            "yolo_ready": False,
            "ocr_ready": False,
            "source_type": "idle",
            "source": "",
            "running": False,
            "message": "后端启动中，正在等待 ONNX 与 OCR 模型加载。",
            "detector_model": "plate",
            "active_model_name": "车牌识别模型",
            "model_label": "plate",
            "model_display_label": "车牌",
            "model_uses_ocr": True,
            "model_class_names": {},
            "model_metadata": {},
            "display_fps": 0.0,
            "capture_fps": 0.0,
            "yolo_fps": 0.0,
            "ocr_fps": 0.0,
            "yolo_ms": 0.0,
            "ocr_ms": 0.0,
            "latency_ms": 0.0,
            "frame_id": 0,
            "motion_score": 0.0,
            "sync_mode": "auto",
            "model_path": str(DEFAULT_ONNX_PATH),
            "model_input_width": 0,
            "model_input_height": 0,
            "model_input_size": 0,
            "config_imgsz": self.config.imgsz,
            "yolo_provider": "unknown",
            "actual_providers": [],
            "available_providers": [],
            "trt_cache_path": "",
            "trt_warmup_ms": 0.0,
            "yolo_pre_ms": 0.0,
            "yolo_infer_ms": 0.0,
            "yolo_post_ms": 0.0,
            "track_ms": 0.0,
            "overlay_ms": 0.0,
            "encode_ms": 0.0,
            "frame_age_ms": 0.0,
            "ocr_queue_size": 0,
            "ocr_pending_tracks": 0,
            "ocr_model": self.config.ocr_model_name,
            "server_overlay": self.config.server_overlay,
            "provider_init_errors": [],
            "session_provider_requested": [],
            "cuda_preload_ok": False,
            "cuda_preload_message": "",
            "cuda_dll_dirs": [],
            "frame_width": 0,
            "frame_height": 0,
            "parking_alerts": 0,
            "parking_active": 0,
            "parking_threshold_s": self.config.parking_violation_seconds,
        }

        self.ort_session: Any = None
        self.ort_input_name: str = ""
        self.ort_output_names: List[str] = []
        self.ort_input_height: int = 0
        self.ort_input_width: int = 0
        self.ort_input_size: int = 0
        self.model_class_names: Dict[int, str] = {}
        self.model_metadata: Dict[str, str] = {}
        self.ocr_model: Any = None

        self.stop_event = threading.Event()
        # 模型切换只暂停 YOLO 推理线程，不停止视频读取线程。
        # 因此手机 VideoCapture 始终保持原连接，切换 best/hearmap/stop/normal 不会重新拉流。
        self.model_switch_event = threading.Event()
        self.model_switch_lock = threading.RLock()
        self.inference_lock = threading.RLock()
        self.model_generation: int = 0
        self.capture_thread: Optional[threading.Thread] = None
        self.yolo_thread: Optional[threading.Thread] = None
        self.ocr_thread: Optional[threading.Thread] = None
        self.ocr_queue: queue.Queue = queue.Queue(maxsize=self.config.max_ocr_queue)

        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_id: int = 0
        self.latest_frame_ts: float = 0.0
        self.source_finished: bool = False
        self.motion_gray: Optional[np.ndarray] = None
        self.last_motion_reset_t: float = 0.0
        self.motion_score: float = 0.0

        self.tracks: List[PlateTrack] = []
        self.next_track_id = 1
        # 只保存“最近一次 YOLO 推理产生的实时框”。
        # 注意：显示层画框只以这里为准，不再等待 OCR，也不再直接画历史 active_tracks。
        self.latest_yolo_boxes: List[Dict[str, Any]] = []
        self.latest_detections: List[Dict[str, Any]] = []
        self.latest_result: Dict[str, Any] = {"plates": [], "detections": [], "tracks": [], "message": "等待启动工作流。"}

        # 正常道路模式专用：用户手动画 ROI 后，车道线算法只在 detector_model=normal 时运行。
        # 不再启动旧版独立 lane_detection_worker，也不会影响其他三个检测模式。
        self.normal_roi_normalized: List[List[float]] = []
        self.normal_lane_detector = NormalLaneDetector()
        self.normal_lane_result: Dict[str, Any] = self.normal_lane_detector.empty_result()

        # 兼容旧字段名；当前版本每次热力图计算都会清空，不再保留历史热点。
        self.heatmap_memory: Dict[str, Dict[str, Any]] = {}

        # 禁停监测状态：ByteTrack/IoU 风格跟踪 car 目标，记录其进入 noParking 区域后的连续停留时间。
        self.parking_tracks: Dict[str, Dict[str, Any]] = {}
        self.parking_zones: Dict[str, Dict[str, Any]] = {}
        self.parking_event_seq: int = 1
        # 永久/半永久禁停事件记录：告警触发后保留在侧边栏，避免车辆离开或漏检后用户看不到“到底哪辆车违停”。
        self.parking_event_history: List[Dict[str, Any]] = []

        # 性能监测：保存滑动窗口耗时与计数，便于定位瓶颈。
        self.perf_samples: Dict[str, Any] = {}
        self.perf_counters: Dict[str, int] = {}
        self.perf_reset_at: float = time.time()
        self.last_perf_log_t: float = 0.0

    def update_status(self, **kwargs: Any) -> None:
        with self.lock:
            self.status.update(kwargs)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            status = dict(self.status)
            result = json.loads(json.dumps(self.latest_result, ensure_ascii=False))
            config = safe_model_dump(self.config)
        return {
            "status": status,
            "config": config,
            "result": result,
            "perf": get_perf_summary(),
        }


STATE = RuntimeState()
app = FastAPI(title="TrafficVision Plate Backend ONNX Parallel", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 旧版独立车道线线程已移除；新版只在 normal.onnx 模式内同步运行。


def db_connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_exec(sql: str, params: tuple = ()) -> None:
    with db_connect() as conn:
        conn.execute(sql, params)
        conn.commit()


def db_query(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    with db_connect() as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    salt = salt or uuid.uuid4().hex[:16]
    digest = hashlib.sha256((salt + str(password)).encode('utf-8')).hexdigest()
    return salt, digest


def verify_password(password: str, salt: str, password_hash: str) -> bool:
    return hash_password(password, salt)[1] == password_hash


def log_operation(action: str, detail: str = "", username: str = "system") -> None:
    try:
        db_exec(
            "INSERT INTO operation_logs(username, action, detail, created_at) VALUES(?,?,?,?)",
            (username, action, detail, time.strftime('%Y-%m-%d %H:%M:%S')),
        )
    except Exception:
        pass


def init_system_database() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with db_connect() as conn:
        cur = conn.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'operator',
            display_name TEXT,
            created_at TEXT,
            last_login TEXT
        );
        CREATE TABLE IF NOT EXISTS video_sources(
            source_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT,
            stream_url TEXT,
            position TEXT,
            status TEXT,
            note TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS model_configs(
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_key TEXT UNIQUE,
            model_name TEXT,
            path TEXT,
            purpose TEXT,
            input_size TEXT,
            provider TEXT,
            class_names TEXT,
            status TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS plate_whitelist(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_no TEXT UNIQUE NOT NULL,
            owner TEXT,
            allow INTEGER DEFAULT 1,
            note TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS system_configs(
            config_key TEXT PRIMARY KEY,
            config_value TEXT,
            description TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS operation_logs(
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            action TEXT,
            detail TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS plate_records(
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_no TEXT,
            track_id TEXT,
            ocr_score REAL,
            whitelist_hit INTEGER,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS traffic_stats(
            stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_count INTEGER,
            heat_level TEXT,
            max_density INTEGER,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS anomaly_events(
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            bbox_json TEXT,
            duration REAL,
            status TEXT,
            created_at TEXT
        );
        """)
        conn.commit()
        if not conn.execute("SELECT 1 FROM users WHERE username='admin'").fetchone():
            salt, digest = hash_password('admin123')
            cur.execute("INSERT INTO users(username,password_hash,salt,role,display_name,created_at) VALUES(?,?,?,?,?,?)",
                        ('admin', digest, salt, 'admin', '系统管理员', time.strftime('%Y-%m-%d %H:%M:%S')))
        if not conn.execute("SELECT 1 FROM video_sources").fetchone():
            now = time.strftime('%Y-%m-%d %H:%M:%S')
            cur.executemany("INSERT INTO video_sources(name,type,stream_url,position,status,note,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)", [
                ('本地视频演示源','local_video','','A/B/C/D 演示位置','ready','用于答辩稳定演示',now,now),
                ('手机 IP Webcam','camera', DEFAULT_CAMERA_URL, '移动采集端','standby','Tailscale 虚拟局域网实时流',now,now),
                ('Python 分析节点','backend','http://127.0.0.1:%s'%BACKEND_PORT,'本机','online','FastAPI + ONNXRuntime',now,now),
            ])
        now = time.strftime('%Y-%m-%d %H:%M:%S')
        default_models = [
            ('plate','best(1).onnx 车牌检测', str(DEFAULT_ONNX_PATH), '车牌框检测 + OCR识别', 'auto/416','CUDA/CPU','plate,carNumber','ready'),
            ('vehicle','hearmap.onnx 车辆检测', str(DEFAULT_VEHICLE_ONNX_PATH), '车辆检测与拥堵热力图', 'auto/640','CUDA/CPU','car,vehicle','ready'),
            ('stop','stop.onnx 禁停检测', str(DEFAULT_STOP_ONNX_PATH), 'car/carNumber/noParking 多类别检测与禁停计时', 'auto','CUDA/CPU','car,carNumber,noParking','ready'),
            ('normal','normal.onnx 正常/异常区域检测', str(DEFAULT_NORMAL_ONNX_PATH), '正常区域/异常区域验证模型', 'auto','CUDA/CPU','normal,abnormal','pending'),
        ]
        for row in default_models:
            cur.execute("INSERT OR REPLACE INTO model_configs(model_key,model_name,path,purpose,input_size,provider,class_names,status,updated_at) VALUES(?,?,?,?,?,?,?,?,?)", (*row, now))
        default_configs = [
            ('parking_violation_seconds','3','车辆确认停止后连续停留 3 秒触发禁停告警'),
            ('heatmap_decay_seconds','0','车辆热力图只按当前检测车辆实时刷新，不保留历史热点或淡出'),
            ('default_provider','CUDAExecutionProvider','ONNXRuntime 优先使用 CUDA，失败时回退 CPU'),
            ('show_server_overlay','false','视频框由前端 Canvas 叠加，后端少做画框和二次编码'),
        ]
        for k,v,d in default_configs:
            cur.execute("INSERT OR IGNORE INTO system_configs(config_key,config_value,description,updated_at) VALUES(?,?,?,?)", (k,v,d,now))
        conn.commit()
    log_operation('init_database', '初始化 SQLite 用户、设备、模型、白名单、配置与日志表')


def safe_model_dump(model: Any) -> Dict[str, Any]:
    return model.model_dump() if hasattr(model, 'model_dump') else model.dict()

def log(msg: str) -> None:
    print(f"[PlateBackend] {msg}", flush=True)


def perf_add(name: str, value_ms: float) -> None:
    """记录一个阶段耗时，单位 ms。"""
    try:
        if not STATE.config.perf_monitor_enabled:
            return
        v = float(value_ms)
        if v < 0 or v != v:
            return
        with STATE.lock:
            dq = STATE.perf_samples.get(name)
            if dq is None:
                dq = deque(maxlen=max(30, int(STATE.config.perf_sample_window)))
                STATE.perf_samples[name] = dq
            dq.append(v)
    except Exception:
        pass


def perf_inc(name: str, n: int = 1) -> None:
    try:
        if not STATE.config.perf_monitor_enabled:
            return
        with STATE.lock:
            STATE.perf_counters[name] = int(STATE.perf_counters.get(name, 0)) + int(n)
    except Exception:
        pass


def _stat(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"last": 0.0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "count": 0}
    arr = sorted(float(v) for v in values)
    n = len(arr)
    p50 = arr[min(n - 1, int(n * 0.50))]
    p95 = arr[min(n - 1, int(n * 0.95))]
    return {
        "last": round(float(values[-1]), 2),
        "avg": round(sum(arr) / max(1, n), 2),
        "p50": round(p50, 2),
        "p95": round(p95, 2),
        "max": round(max(arr), 2),
        "count": n,
    }


def get_perf_summary() -> Dict[str, Any]:
    with STATE.lock:
        samples = {k: list(v) for k, v in STATE.perf_samples.items()}
        counters = dict(STATE.perf_counters)
        status = dict(STATE.status)
        latest_frame_ts = float(STATE.latest_frame_ts or 0.0)
        latest_id = int(STATE.latest_frame_id)
        tracks = list(STATE.tracks)
        qsize = int(STATE.ocr_queue.qsize()) if STATE.ocr_queue is not None else 0
        reset_at = float(STATE.perf_reset_at)
    stage_ms = {k: _stat(v) for k, v in samples.items()}
    pending = sum(1 for t in tracks if getattr(t, "ocr_pending", False))
    frame_age_ms = round((time.time() - latest_frame_ts) * 1000.0, 2) if latest_frame_ts else 0.0
    bottlenecks: List[str] = []
    provider = str(status.get("yolo_provider", "unknown"))
    if "CPU" in provider:
        bottlenecks.append("当前 ONNX Session 实际落在 CPU，优先检查 TensorRT/CUDA Provider 是否初始化失败。")
    yolo = stage_ms.get("yolo_total_ms", {}).get("avg", 0.0)
    infer = stage_ms.get("yolo_infer_ms", {}).get("avg", 0.0)
    pre = stage_ms.get("yolo_pre_ms", {}).get("avg", 0.0)
    post = stage_ms.get("yolo_post_ms", {}).get("avg", 0.0)
    enc = stage_ms.get("jpeg_encode_ms", {}).get("avg", 0.0)
    overlay = stage_ms.get("overlay_ms", {}).get("avg", 0.0)
    if infer and yolo and infer / max(yolo, 1e-6) > 0.70:
        bottlenecks.append("YOLO 耗时主要集中在 GPU/ONNX 推理；可考虑 TensorRT engine cache、FP16、固定输入尺寸 warmup。")
    if pre and yolo and pre / max(yolo, 1e-6) > 0.25:
        bottlenecks.append("YOLO 前处理占比偏高；后续可考虑复用输入缓冲、ROI 搜索、减少重复 resize/copy。")
    if post and yolo and post / max(yolo, 1e-6) > 0.25:
        bottlenecks.append("YOLO 后处理/NMS 占比偏高；后续可考虑向量化后处理或导出带 NMS 的 ONNX/TensorRT。")
    if enc > 8.0 or (yolo and enc > yolo * 0.55):
        bottlenecks.append("JPEG 编码/后端画面输出耗时明显；后续可考虑前端 Canvas 叠框，让后端少做画框和二次编码。")
    if overlay > 8.0:
        bottlenecks.append("overlay 绘制耗时偏高；可能与中文绘制/PIL 转换有关，建议前端 Canvas 叠加文字。")
    if counters.get("ocr_queue_drop", 0) > 0:
        bottlenecks.append("OCR 队列出现丢弃，说明 OCR 通道跟不上；应继续做 ROI 质量筛选和低优先级 OCR 调度。")
    if frame_age_ms > 300:
        bottlenecks.append("最新帧年龄偏大，可能是拉流或显示链路滞后，需要检查 cap.read 与网络流缓存。")
    if not bottlenecks:
        bottlenecks.append("暂未发现单点瓶颈，建议连续手持测试 30 秒后查看 p95 耗时。")
    return {
        "stage_ms": stage_ms,
        "counters": counters,
        "queue": {"ocr_size": qsize, "ocr_pending_tracks": pending},
        "frame": {"latest_frame_id": latest_id, "frame_age_ms": frame_age_ms},
        "uptime_sec": round(time.time() - reset_at, 2),
        "bottlenecks": bottlenecks[:6],
    }


def maybe_log_perf() -> None:
    try:
        interval = float(STATE.config.perf_log_interval)
        if interval <= 0:
            return
        now = time.time()
        with STATE.lock:
            last = float(STATE.last_perf_log_t or 0.0)
            if now - last < interval:
                return
            STATE.last_perf_log_t = now
        perf = get_perf_summary()
        yolo = perf["stage_ms"].get("yolo_total_ms", {})
        enc = perf["stage_ms"].get("jpeg_encode_ms", {})
        ocr = perf["stage_ms"].get("ocr_recognize_ms", {})
        log(f"PERF yolo_avg={yolo.get('avg',0)}ms yolo_p95={yolo.get('p95',0)}ms encode_avg={enc.get('avg',0)}ms ocr_avg={ocr.get('avg',0)}ms q={perf['queue'].get('ocr_size',0)} provider={STATE.status.get('yolo_provider')}")
    except Exception:
        pass


def find_chinese_font() -> Optional[str]:
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


_FONT_PATH = find_chinese_font()


def draw_label(frame: np.ndarray, text: str, xy: Tuple[int, int], color: Tuple[int, int, int] = (0, 255, 120)) -> np.ndarray:
    if not text:
        return frame
    x, y = xy
    if Image is not None and ImageDraw is not None and ImageFont is not None:
        try:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            font = ImageFont.truetype(_FONT_PATH, 22) if _FONT_PATH else ImageFont.load_default()
            bbox = draw.textbbox((x, y), text, font=font)
            draw.rectangle((bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3), fill=(8, 18, 32))
            draw.text((x, y), text, fill=(color[2], color[1], color[0]), font=font)
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    cv2.putText(frame, text.encode("ascii", "ignore").decode("ascii") or "PLATE", (x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return frame



def resolve_onnx_input_hw(session: Any, fallback_imgsz: int) -> Tuple[int, int]:
    """读取当前 ONNX 模型真实输入尺寸。

    best(1).onnx 当前按 416 运行；hearmap.onnx 的错误日志显示它固定要求 640×640。
    这里不再硬编码 cfg.imgsz，而是优先从 session.get_inputs()[0].shape 读取 H/W。
    如果模型是动态输入，才回退到配置里的 fallback_imgsz。
    """
    fallback = int(fallback_imgsz or 416)
    try:
        inp = session.get_inputs()[0]
        shape = list(getattr(inp, "shape", []) or [])
        h_raw = shape[2] if len(shape) >= 4 else fallback
        w_raw = shape[3] if len(shape) >= 4 else fallback

        def _as_dim(v: Any) -> int:
            if isinstance(v, (int, np.integer)) and int(v) > 0:
                return int(v)
            try:
                iv = int(v)
                return iv if iv > 0 else fallback
            except Exception:
                return fallback

        h = _as_dim(h_raw)
        w = _as_dim(w_raw)
        if h != w:
            side = max(h, w)
            return side, side
        return h, w
    except Exception:
        return fallback, fallback


def current_model_input_size() -> int:
    """返回当前模型实际使用的方形输入尺寸。"""
    with STATE.lock:
        size = int(getattr(STATE, "ort_input_size", 0) or 0)
        cfg_size = int(getattr(STATE.config, "imgsz", 416) or 416)
    return size if size > 0 else cfg_size




# ==================== 车辆框直投热力图 / bbox density heatmap ====================
# V4 说明：这里不再假装识别道路白线，也不再把车辆强行吸附到几条抽象直线。
# 原理改为：hearmap.onnx 检测车辆 bbox -> 取车辆“落地点/中心点” -> 直接生成热力点。
# 这样左右并排、上下同线、斜向分布都会按照真实框位置显示，不会被道路线归类误差拉偏。

HEATMAP_GRID_COLS = 12
HEATMAP_GRID_ROWS = 8
HEATMAP_DECAY = 0.0
HEATMAP_KERNEL_SCALE = 1.0


def _norm_clamp(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def assign_bbox_to_road(bbox: Any, frame_w: int, frame_h: int) -> Dict[str, Any]:
    """兼容旧字段名：实际返回的是车辆框热力点，不再返回道路归属。"""
    try:
        x1, y1, x2, y2 = [float(v) for v in list(bbox)[:4]]
        fw = max(1.0, float(frame_w or 1))
        fh = max(1.0, float(frame_h or 1))
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        # 车辆热力建议用 bottom-center 作为落地点，比 bbox 几何中心更贴近道路平面；
        # 同时保留几何中心用于前端调试。
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        anchor_x = cx
        anchor_y = y2

        cxn = _norm_clamp(cx / fw)
        cyn = _norm_clamp(cy / fh)
        axn = _norm_clamp(anchor_x / fw)
        ayn = _norm_clamp(anchor_y / fh)
        bw_norm = _norm_clamp(bw / fw)
        bh_norm = _norm_clamp(bh / fh)

        # 热力半径跟车辆框大小相关，但做上下限，避免近处大车把整张图盖住。
        radius_norm = max(0.035, min(0.115, (bw_norm + bh_norm) * 0.62))

        return {
            "road_id": "bbox_heat",
            "road_name": "车辆热力点",
            "lane": "bbox",
            "segment": "实时点位",
            "road_x_norm": round(axn, 4),
            "center_x": round(cx, 2),
            "center_y": round(cy, 2),
            "center_x_norm": round(cxn, 4),
            "center_y_norm": round(cyn, 4),
            "anchor_x_norm": round(axn, 4),
            "anchor_y_norm": round(ayn, 4),
            "bbox_w_norm": round(bw_norm, 4),
            "bbox_h_norm": round(bh_norm, 4),
            "radius_norm": round(radius_norm, 4),
            "lateral_distance_norm": 0.0,
        }
    except Exception:
        return {
            "road_id": "bbox_heat",
            "road_name": "车辆热力点",
            "lane": "bbox",
            "segment": "未知",
            "road_x_norm": 0.5,
            "center_x": 0,
            "center_y": 0,
            "center_x_norm": 0.5,
            "center_y_norm": 0.5,
            "anchor_x_norm": 0.5,
            "anchor_y_norm": 0.5,
            "bbox_w_norm": 0.05,
            "bbox_h_norm": 0.05,
            "radius_norm": 0.06,
            "lateral_distance_norm": 0.0,
        }


def _grid_cell_for_point(x_norm: float, y_norm: float) -> str:
    col = int(max(0, min(HEATMAP_GRID_COLS - 1, int(float(x_norm) * HEATMAP_GRID_COLS))))
    row = int(max(0, min(HEATMAP_GRID_ROWS - 1, int(float(y_norm) * HEATMAP_GRID_ROWS))))
    return f"G{row + 1}-{col + 1}"


HEATMAP_MEMORY_TTL = 0.0
HEATMAP_MEMORY_KEEP_TTL = 0.0


def _make_heatmap_key(box: Dict[str, Any]) -> str:
    track_id = str(box.get("track_id") or "")
    if track_id:
        return track_id
    track_num = str(box.get("track_num") or "")
    if track_num:
        return f"track-{track_num}"
    bbox = box.get("bbox") or [0, 0, 0, 0]
    try:
        x1, y1, x2, y2 = [int(float(v)) for v in list(bbox)[:4]]
        return f"bbox-{x1//20}-{y1//20}-{x2//20}-{y2//20}"
    except Exception:
        return f"unknown-{time.time():.3f}"


def update_heatmap_memory(boxes: List[Dict[str, Any]], frame_w: int, frame_h: int) -> List[Dict[str, Any]]:
    """兼容旧函数名，但不再保存历史热点。

    热力图严格由本次 YOLO 输出的车辆数量和位置生成：本次未检出车辆时立即返回空点集，
    不做 3 秒淡出、不做惯性预测，也不保留上一帧车辆。
    """
    valid_boxes = [b for b in (boxes or []) if is_heatmap_vehicle_detection(b)]
    points: List[Dict[str, Any]] = []

    for index, box in enumerate(valid_boxes):
        point = assign_bbox_to_road(box.get("bbox", []), frame_w, frame_h)
        conf = max(0.0, min(1.0, float(box.get("det_confidence", box.get("confidence", 0.0)) or 0.0)))
        ax = float(point.get("anchor_x_norm", point.get("center_x_norm", 0.5)) or 0.5)
        ay = float(point.get("anchor_y_norm", point.get("center_y_norm", 0.5)) or 0.5)
        key = _make_heatmap_key(box) or f"current-{index}"
        point.update({
            "track_id": box.get("track_id", key),
            "track_num": box.get("track_num", ""),
            "confidence": round(conf, 4),
            "weight": round(max(0.05, conf), 4),
            "bbox": box.get("bbox", []),
            "cell_id": _grid_cell_for_point(ax, ay),
            "memory_age": 0.0,
            "memory_decay": 1.0,
            "semantic_type": "vehicle",
        })
        points.append(point)

    with STATE.lock:
        # 明确清空旧版记忆，切换到新逻辑后不会残留历史热点。
        STATE.heatmap_memory = {}
    return points


def build_abstract_road_map(boxes: List[Dict[str, Any]], frame_w: int, frame_h: int) -> Dict[str, Any]:
    """根据当前车辆检测框生成实时热力图数据。

    注意：这里严格只统计 semantic_type=vehicle/car 的目标；noParking、carNumber、normal 不进入热力图。
    """
    points: List[Dict[str, Any]] = update_heatmap_memory(boxes, frame_w, frame_h)
    cell_counter: Dict[str, int] = {}
    for p in points:
        cid = str(p.get("cell_id") or _grid_cell_for_point(float(p.get("anchor_x_norm", 0.5)), float(p.get("anchor_y_norm", 0.5))))
        cell_counter[cid] = cell_counter.get(cid, 0) + 1

    # 给每个点补充局部密度分数：附近车越多，热力越高。
    for i, p in enumerate(points):
        px = float(p.get("anchor_x_norm", 0.5))
        py = float(p.get("anchor_y_norm", 0.5))
        local = 0.0
        for q in points:
            qx = float(q.get("anchor_x_norm", 0.5))
            qy = float(q.get("anchor_y_norm", 0.5))
            dx = px - qx
            dy = py - qy
            d2 = dx * dx + dy * dy
            if d2 < 0.035:
                local += max(0.25, float(q.get("weight", 0.5))) * (1.0 / (1.0 + d2 * 55.0))
        p["density_score"] = round(max(0.0, min(1.0, local / 2.4)), 4)
        p["rank"] = i + 1

    heat = []
    max_count = max([1] + list(cell_counter.values()))
    for row in range(HEATMAP_GRID_ROWS):
        for col in range(HEATMAP_GRID_COLS):
            cid = f"G{row + 1}-{col + 1}"
            count = int(cell_counter.get(cid, 0))
            if count <= 0:
                continue
            heat.append({
                "road_id": cid,
                "road_name": f"区域 {row + 1}-{col + 1}",
                "lane": "grid",
                "count": count,
                "heat_score": round(count / max_count, 4),
                "x_norm": round((col + 0.5) / HEATMAP_GRID_COLS, 4),
                "y_norm": round((row + 0.5) / HEATMAP_GRID_ROWS, 4),
            })

    vehicle_count = len(points)
    avg_conf = round(sum(float(p.get("confidence", 0)) for p in points) / max(1, vehicle_count), 4) if vehicle_count else 0.0
    max_cell_count = max([0] + [int(h.get("count", 0)) for h in heat])

    # 高级热力图需要不仅知道“点在哪里”，还要知道局部团簇、热区中心和强度。
    # 这里仍然不识别道路白线，而是对车辆落地点做空间核密度估计的前置统计。
    hotspots = []
    sorted_points = sorted(points, key=lambda p: float(p.get("density_score", 0)), reverse=True)
    used = []
    for p in sorted_points:
        px = float(p.get("anchor_x_norm", 0.5))
        py = float(p.get("anchor_y_norm", 0.5))
        if any(((px - ux) ** 2 + (py - uy) ** 2) < 0.012 for ux, uy in used):
            continue
        cluster = []
        for q in points:
            qx = float(q.get("anchor_x_norm", 0.5))
            qy = float(q.get("anchor_y_norm", 0.5))
            d2 = (px - qx) ** 2 + (py - qy) ** 2
            if d2 < 0.035:
                cluster.append(q)
        if not cluster:
            continue
        heat_score = min(1.0, sum(float(q.get("weight", 0.45)) for q in cluster) / 3.0)
        hotspots.append({
            "x_norm": round(px, 4),
            "y_norm": round(py, 4),
            "count": len(cluster),
            "heat_score": round(heat_score, 4),
            "track_ids": [q.get("track_id", "") for q in cluster[:6]],
        })
        used.append((px, py))
        if len(hotspots) >= 5:
            break

    if vehicle_count >= 6 or max_cell_count >= 3:
        level = "high"
        label = "高密度"
    elif vehicle_count >= 3 or max_cell_count >= 2:
        level = "medium"
        label = "中等密度"
    elif vehicle_count > 0:
        level = "low"
        label = "低密度"
    else:
        level = "waiting"
        label = "等待车辆模型"

    return {
        "mode": "bbox_kde_heatmap_v2",
        "description": "当前帧车辆检测框直接生成高级核密度热力图：不保留历史热点，车辆数量与位置随本次检测实时更新。",
        "frame_width": int(frame_w or 0),
        "frame_height": int(frame_h or 0),
        "roads": [],
        "points": points,
        "assignments": points,  # 兼容前端旧字段名
        "heat": heat,
        "hotspots": hotspots,
        "algorithm": {
            "name": "bbox-kde-direct",
            "anchor": "bottom_center",
            "grid_cols": HEATMAP_GRID_COLS,
            "grid_rows": HEATMAP_GRID_ROWS,
            "decay": 0.0,
            "history_memory": False,
            "realtime_only": True,
            "kernel_scale": HEATMAP_KERNEL_SCALE,
            "road_line_recognition": False,
        },
        "summary": {
            "vehicle_count": vehicle_count,
            "avg_confidence": avg_conf,
            "max_cell_count": max_cell_count,
            "density_level": level,
            "density_label": label,
            "grid_cols": HEATMAP_GRID_COLS,
            "grid_rows": HEATMAP_GRID_ROWS,
        },
    }



def _parse_class_names_value(raw: Any) -> Dict[int, str]:
    """
    兼容 Ultralytics/YOLO ONNX 元数据里的 names/classes/labels。
    常见格式：
    - "{0: 'car', 1: 'stop', 2: 'normal'}"
    - "['car', 'bus', 'truck']"
    - '{"0":"car","1":"bus"}'
    """
    if raw is None:
        return {}
    value = raw
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return {}
        parsed = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(txt)
                break
            except Exception:
                parsed = None
        if parsed is None:
            # 支持 "car,bus,truck" 这类简写
            if "," in txt:
                return {i: x.strip() for i, x in enumerate(txt.split(",")) if x.strip()}
            return {}
        value = parsed

    names: Dict[int, str] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            try:
                idx = int(k)
            except Exception:
                continue
            name = str(v).strip()
            if name:
                names[idx] = name
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            name = str(v).strip()
            if name:
                names[int(i)] = name
    return names


def extract_onnx_class_names(sess: Any, model_key: str = "") -> Tuple[Dict[int, str], Dict[str, str]]:
    """从 ONNX metadata 中提取真实类别名，避免 stop/normal 多类别被统一显示成一个中文总名。"""
    metadata: Dict[str, str] = {}
    names: Dict[int, str] = {}
    try:
        meta = sess.get_modelmeta()
        raw_map = getattr(meta, "custom_metadata_map", {}) or {}
        metadata = {str(k): str(v) for k, v in raw_map.items()}
        # 常见 key 优先级
        candidate_keys = ["names", "classes", "labels", "class_names", "categories"]
        for key in candidate_keys:
            if key in raw_map:
                names = _parse_class_names_value(raw_map.get(key))
                if names:
                    break
        # 有些模型会把 names 放在非标准 key 中，兜底扫一遍
        if not names:
            for k, v in raw_map.items():
                lk = str(k).lower()
                if "name" in lk or "class" in lk or "label" in lk:
                    names = _parse_class_names_value(v)
                    if names:
                        break
    except Exception as exc:
        metadata = {"metadata_error": repr(exc)}
        names = {}

    # 允许用户用环境变量临时覆盖：STOP_CLASS_NAMES="xxx,yyy,zzz"
    env_key = f"{str(model_key or '').upper()}_CLASS_NAMES"
    env_names = _parse_class_names_value(os.environ.get(env_key, ""))
    if env_names:
        names = env_names
        metadata[env_key] = os.environ.get(env_key, "")
    return names, metadata


def class_display_name(model_info: Dict[str, Any], class_id: int, class_names: Optional[Dict[int, str]] = None) -> str:
    """返回每个检测框自己的原始类别名；没有 metadata 时也要保留类别编号，不能全显示成一个总名。"""
    names = class_names if class_names is not None else getattr(STATE, "model_class_names", {})
    try:
        cid = int(class_id)
    except Exception:
        cid = 0
    if names and cid in names:
        return str(names[cid])
    key = str(model_info.get("key", ""))
    base = str(model_info.get("display_label", "目标"))
    if key in {"plate", "vehicle"} and cid == 0:
        return base
    return f"{base}类别{cid}"


def _canonical_class_name(name: Any) -> str:
    """把模型里的 class 名统一成可做业务判断的英文小写键。"""
    text = str(name or "").strip()
    text = text.replace(" ", "").replace("_", "").replace("-", "").lower()
    text = text.replace("停车", "parking").replace("禁停", "noparking").replace("车牌", "carnumber").replace("车辆", "car")
    return text


def semantic_class_type(model_key: str, class_id: int, class_name: str = "") -> str:
    """统一不同模型的标签语义。

    解决的问题：
    - stop.onnx / normal.onnx 里可能同时输出 car、carNumber、noParking；
    - 热力图只能统计 car/vehicle，不能把 noParking 当车辆热点；
    - carNumber 要和车牌检测含义统一，必要时走 OCR。
    """
    key = normalize_detector_model(model_key)
    c = _canonical_class_name(class_name)
    try:
        cid = int(class_id)
    except Exception:
        cid = 0

    if c in {"car", "vehicle", "cars", "auto", "automobile"}:
        return "vehicle"
    if c in {"bus", "truck", "van"}:
        return "vehicle"
    if c in {"carnumber", "licenseplate", "plate", "platenumber", "lp", "numberplate"}:
        return "plate_number"
    if c in {"noparking", "nopark", "stop", "stopzone", "forbiddenparking", "parkingforbidden", "nostop", "noStopping".lower()} or "noparking" in c:
        return "no_parking"
    if c in {"normal", "normalzone", "normalarea", "road", "free", "allowed"}:
        return "normal_zone"

    # 没有类别名元数据时，按当前模型做最保守兜底。
    if key == "vehicle":
        return "vehicle" if cid == 0 else "vehicle_other"
    if key == "plate":
        return "plate_number"
    if key == "stop":
        return "unknown_zone"
    if key == "normal":
        return "unknown_zone"
    return "unknown"


def unified_display_name(model_key: str, class_id: int, class_name: str = "", fallback: str = "") -> str:
    sem = semantic_class_type(model_key, class_id, class_name)
    if sem == "vehicle":
        return "车辆"
    if sem == "plate_number":
        return "车牌区域"
    if sem == "no_parking":
        return "禁停区域"
    if sem == "normal_zone":
        return "正常区域"
    return str(fallback or class_name or "目标")


def is_heatmap_vehicle_detection(item: Dict[str, Any]) -> bool:
    """热力图只允许车辆/car 类进入，noParking/carNumber/normal 一律不作为热点。"""
    sem = str(item.get("semantic_type") or "")
    if sem:
        return sem == "vehicle"
    return semantic_class_type(str(item.get("detector_model", "")), int(item.get("class_id", 0) or 0), str(item.get("class_name", ""))) == "vehicle"


def is_plate_like_detection(item: Dict[str, Any], model_key: str = "") -> bool:
    sem = str(item.get("semantic_type") or "")
    if sem:
        return sem == "plate_number"
    return semantic_class_type(model_key or str(item.get("detector_model", "")), int(item.get("class_id", 0) or 0), str(item.get("class_name", ""))) == "plate_number"


def is_no_parking_detection(item: Dict[str, Any], model_key: str = "") -> bool:
    sem = str(item.get("semantic_type") or "")
    if sem:
        return sem == "no_parking"
    return semantic_class_type(model_key or str(item.get("detector_model", "")), int(item.get("class_id", 0) or 0), str(item.get("class_name", ""))) == "no_parking"


def _bbox_area_xyxy(bbox: Any) -> float:
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
    except Exception:
        return 0.0


def _bbox_intersection_area(a: Any, b: Any) -> float:
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    except Exception:
        return 0.0


def _point_in_bbox(px: float, py: float, bbox: Any) -> bool:
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return x1 <= px <= x2 and y1 <= py <= y2
    except Exception:
        return False


def _parking_zone_key(zone_box: Dict[str, Any]) -> str:
    bbox = zone_box.get("bbox") or [0, 0, 0, 0]
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx = int(round((x1 + x2) / 80.0))
        cy = int(round((y1 + y2) / 80.0))
        cid = int(zone_box.get("class_id", 0) or 0)
        return f"z{cid}_{cx}_{cy}"
    except Exception:
        return f"zone_{int(time.time() * 1000)}"


def _parking_relation_score(car_bbox: Any, zone_bbox: Any, cfg: BackendConfig) -> Tuple[bool, float, str]:
    try:
        x1, y1, x2, y2 = [float(v) for v in car_bbox]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bottom_x = cx
        bottom_y = y2
        if _point_in_bbox(cx, cy, zone_bbox):
            return True, 1.0, "center_inside"
        if _point_in_bbox(bottom_x, bottom_y, zone_bbox):
            return True, 0.92, "bottom_inside"
        inter = _bbox_intersection_area(car_bbox, zone_bbox)
        car_area = max(1.0, _bbox_area_xyxy(car_bbox))
        ratio = inter / car_area
        if ratio >= max(0.01, float(cfg.parking_intersection_ratio)):
            return True, min(0.9, 0.35 + ratio), "intersection"
        return False, ratio, "outside"
    except Exception:
        return False, 0.0, "error"


def _parking_vehicle_anchor(bbox: Any) -> Tuple[float, float]:
    """车辆跟踪/速度判断锚点。

    采用 bbox 底部中心点，比中心点更接近车辆与道路接触位置；禁停计时的
    “是否停止”也以这个点的运动速度为准。
    """
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return (x1 + x2) / 2.0, y2
    except Exception:
        return 0.0, 0.0


def _bbox_overlap_ratio_to_small(inner_bbox: Any, outer_bbox: Any) -> float:
    """inner 与 outer 的交叠面积 / inner 面积，用来判断 carNumber 是否已经被 car 框覆盖。"""
    try:
        inter = _bbox_intersection_area(inner_bbox, outer_bbox)
        return inter / max(1.0, _bbox_area_xyxy(inner_bbox))
    except Exception:
        return 0.0




def _bbox_center_xy(bbox: Any) -> Tuple[float, float]:
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    except Exception:
        return 0.0, 0.0


def _expand_bbox_xyxy(bbox: Any, sx: float = 1.75, sy: float = 2.25) -> List[float]:
    """以 bbox 中心向外扩张，专门用于把 carNumber 归并到同一辆 car。"""
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        return [cx - w * sx / 2.0, cy - h * sy / 2.0, cx + w * sx / 2.0, cy + h * sy / 2.0]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]


def _plate_related_to_car(plate_box: Dict[str, Any], car_box: Dict[str, Any]) -> bool:
    """判断 carNumber 是否属于某个 car。

    旧版只用“车牌框面积交叠比例”判断，车牌框很小或刚好在车辆框边缘时容易漏掉，
    导致同一辆车在侧边栏同时出现“车辆框”和“车牌代理”两条记录。
    这里改成：交叠 + 扩展车辆框包含车牌中心 + 中心距离 三重判断。
    """
    pb = plate_box.get('bbox') or []
    cb = car_box.get('bbox') or []
    if not pb or not cb:
        return False
    try:
        # 车牌框只要与车辆框有一点点交叠，大概率就是同一辆车。
        if _bbox_overlap_ratio_to_small(pb, cb) >= 0.03:
            return True
        pcx, pcy = _bbox_center_xy(pb)
        ec = _expand_bbox_xyxy(cb, 1.85, 2.35)
        if _point_in_bbox(pcx, pcy, ec):
            return True
        ccx, ccy = _bbox_center_xy(cb)
        x1, y1, x2, y2 = [float(v) for v in cb]
        cw, ch = max(1.0, x2 - x1), max(1.0, y2 - y1)
        # 对沙盘小车而言，车牌框和车辆框中心距离不会超过车辆自身尺度太多。
        if math.hypot(pcx - ccx, pcy - ccy) <= max(cw, ch) * 1.10:
            return True
    except Exception:
        return False
    return False


def _best_plate_for_car(car_box: Dict[str, Any], plate_boxes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best = None
    best_score = -1.0
    cb = car_box.get('bbox') or []
    if not cb:
        return None
    ccx, ccy = _bbox_center_xy(cb)
    for p in plate_boxes or []:
        if not _plate_related_to_car(p, car_box):
            continue
        pcx, pcy = _bbox_center_xy(p.get('bbox') or [])
        overlap = _bbox_overlap_ratio_to_small(p.get('bbox') or [], cb)
        dist = math.hypot(pcx - ccx, pcy - ccy)
        score = overlap * 10.0 - dist / 1000.0 + float(p.get('det_confidence', p.get('confidence', 0.0)) or 0.0)
        if score > best_score:
            best_score = score
            best = p
    return dict(best) if best is not None else None


def _parking_track_display_name(item: Dict[str, Any]) -> str:
    """禁停侧边栏/历史记录中展示“到底是哪辆车”。"""
    plate_text = str(item.get('associated_plate_text') or item.get('plate_text') or item.get('stable_text') or '').strip()
    if plate_text:
        return plate_text
    plate_track = str(item.get('associated_plate_track_id') or '').strip()
    if plate_track:
        return f"车辆{item.get('track_id', '')} / {plate_track}"
    return str(item.get('track_id') or f"VEH-{int(item.get('track_num', 0) or 0):03d}")


def _append_parking_event_history(event_item: Dict[str, Any]) -> None:
    """记录禁停告警历史，并写入 runtime_data/parking_events.jsonl。"""
    event_id = str(event_item.get('event_id') or '')
    if not event_id:
        return
    hist = list(getattr(STATE, 'parking_event_history', []) or [])
    for old in hist:
        if str(old.get('event_id') or '') == event_id:
            old['last_update_at'] = time.time()
            old['dwell_s'] = round(float(event_item.get('dwell_s', old.get('dwell_s', 0.0)) or 0.0), 2)
            STATE.parking_event_history = hist[-80:]
            return
    now_ts = time.time()
    record = {
        'event_id': event_id,
        'created_at': now_ts,
        'created_time': time.strftime('%H:%M:%S', time.localtime(now_ts)),
        'vehicle_name': _parking_track_display_name(event_item),
        'track_id': str(event_item.get('track_id') or ''),
        'track_num': int(event_item.get('track_num', 0) or 0),
        'associated_plate_track_id': str(event_item.get('associated_plate_track_id') or ''),
        'associated_plate_text': str(event_item.get('associated_plate_text') or ''),
        'source_label': str(event_item.get('source_label') or ''),
        'proxy_from': str(event_item.get('proxy_from') or 'car'),
        'zone_label': str(event_item.get('zone_label') or '禁停区域'),
        'dwell_s': round(float(event_item.get('dwell_s', 0.0) or 0.0), 2),
        'threshold_s': round(float(event_item.get('threshold_s', 3.0) or 3.0), 2),
        'status': '已告警',
    }
    hist.append(record)
    STATE.parking_event_history = hist[-80:]
    try:
        PARKING_EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PARKING_EVENT_LOG_PATH.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception as exc:
        log(f'禁停事件写入失败: {exc}')

def _make_plate_proxy_vehicle(plate_box: Dict[str, Any]) -> Dict[str, Any]:
    """当 stop.onnx 只检出 carNumber、没有检出 car 时，用车牌框生成车辆代理。

    注意：这不是把车牌当成热力图车辆目标，而是只用于禁停跟踪侧边栏，解决
    “车在禁停区里但本帧 car 漏检，界面完全没有计时器”的问题。
    """
    p = dict(plate_box or {})
    try:
        tn = int(p.get("track_num", 0) or 0)
    except Exception:
        tn = 0
    if tn <= 0:
        # 没有稳定 track_num 时，用 bbox 粗略生成一个稳定 key，避免完全丢失侧边栏状态。
        try:
            x1, y1, x2, y2 = [float(v) for v in (p.get("bbox") or [0, 0, 0, 0])]
            tn = 8000 + int(round(((x1 + x2) / 2.0) / 16.0)) * 100 + int(round(((y1 + y2) / 2.0) / 16.0))
        except Exception:
            tn = 8999
    proxy = dict(p)
    proxy["semantic_type"] = "vehicle_proxy"
    proxy["parking_track_key"] = f"P{tn}"
    proxy["track_num"] = tn
    proxy["track_id"] = p.get("track_id") or f"PLATE-P{tn:03d}"
    proxy["source_label"] = "车牌代理"
    proxy["proxy_from"] = "plate_number"
    proxy["heatmap_eligible"] = False
    proxy["class_display_name"] = p.get("class_display_name") or p.get("display_label") or "车牌区域"
    proxy["display_label"] = proxy["class_display_name"]
    return proxy


def _build_parking_candidates(boxes: List[Dict[str, Any]], cfg: BackendConfig) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """返回用于禁停计时的目标。

    核心修复：同一辆车同时检出 car 和 carNumber 时，carNumber 只作为 car 的附属证据，
    不再单独生成一条“车牌代理”计时记录。只有当本帧没有任何 car 能和该 carNumber 关联时，
    才启用车牌代理，避免一辆车在侧边栏出现两条。
    """
    raw_cars = [dict(b) for b in (boxes or []) if is_heatmap_vehicle_detection(b)]
    plate_boxes = [dict(b) for b in (boxes or []) if is_plate_like_detection(b)]
    candidates: List[Dict[str, Any]] = []
    used_plate_indexes = set()

    for c in raw_cars:
        cc = dict(c)
        try:
            tn = int(cc.get("track_num", 0) or 0)
        except Exception:
            tn = 0
        if tn > 0:
            cc["parking_track_key"] = f"C{tn}"
        cc.setdefault("source_label", "车辆框")
        cc.setdefault("proxy_from", "car")

        best_plate = _best_plate_for_car(cc, plate_boxes)
        if best_plate:
            try:
                pidx = next((i for i, p in enumerate(plate_boxes) if p is best_plate or (p.get('track_id') == best_plate.get('track_id') and p.get('bbox') == best_plate.get('bbox'))), -1)
                if pidx >= 0:
                    used_plate_indexes.add(pidx)
            except Exception:
                pass
            cc["associated_plate_track_id"] = str(best_plate.get("track_id") or "")
            cc["associated_plate_track_num"] = int(best_plate.get("track_num", 0) or 0)
            cc["associated_plate_text"] = str(best_plate.get("plate_text") or best_plate.get("stable_text") or "")
            cc["associated_plate_bbox"] = best_plate.get("bbox") or []
            cc["source_label"] = "车辆框+车牌"
        candidates.append(cc)

    proxy_count = 0
    if bool(getattr(cfg, "parking_plate_proxy_enabled", True)):
        for idx, p in enumerate(plate_boxes):
            if idx in used_plate_indexes:
                continue
            # 只要和任意 car 有关系，就不要生成单独代理；它已经属于车辆框。
            if any(_plate_related_to_car(p, c) for c in raw_cars):
                continue
            candidates.append(_make_plate_proxy_vehicle(p))
            proxy_count += 1
    return candidates, len(raw_cars), len(plate_boxes), proxy_count

def _parking_motion_state(old: Dict[str, Any], car_bbox: Any, now: float, frame_w: int, frame_h: int, cfg: BackendConfig) -> Tuple[float, float, bool, float, float]:
    """计算车辆是否已经停止。

    返回：raw_speed_norm、ema_speed_norm、stationary、stationary_since、last_moving_at。
    speed_norm = 像素位移 / 时间 / 画面对角线，能同时适配 720P/1080P。
    """
    ax, ay = _parking_vehicle_anchor(car_bbox)
    last_anchor = old.get("last_anchor") if old else None
    last_ts = float(old.get("last_motion_ts", old.get("last_seen_at", now)) or now) if old else now
    diag = max(1.0, math.hypot(float(frame_w or 0), float(frame_h or 0)))
    dt = max(1e-3, now - last_ts)
    raw_speed = 0.0
    if isinstance(last_anchor, (list, tuple)) and len(last_anchor) >= 2:
        dx = ax - float(last_anchor[0])
        dy = ay - float(last_anchor[1])
        raw_speed = math.hypot(dx, dy) / dt / diag
    old_ema = float(old.get("speed_ema", raw_speed) or 0.0) if old else raw_speed
    # EMA：既能过滤检测框抖动，又能对真实移动快速响应。
    ema_speed = old_ema * 0.65 + raw_speed * 0.35
    speed_limit = float(cfg.parking_stationary_speed_norm)
    is_low_speed = ema_speed <= speed_limit

    prev_stationary_since = float(old.get("stationary_since", 0.0) or 0.0) if old else 0.0
    prev_last_moving_at = float(old.get("last_moving_at", 0.0) or 0.0) if old else 0.0
    if is_low_speed:
        stationary_since = prev_stationary_since or now
        last_moving_at = prev_last_moving_at
    else:
        stationary_since = 0.0
        last_moving_at = now

    stationary_confirmed = bool(stationary_since and (now - stationary_since) >= float(cfg.parking_stationary_confirm_seconds))
    return raw_speed, ema_speed, stationary_confirmed, stationary_since, last_moving_at


def update_parking_monitor(boxes: List[Dict[str, Any]], frame_w: int, frame_h: int) -> Dict[str, Any]:
    """禁停区 ByteTrack/IoU 风格跟踪 + 停止后计时。

    关键原则：
    - car 只作为车辆目标；noParking 只作为禁停 ROI。
    - carNumber 默认不进入热力图；但如果 stop.onnx 偶发漏掉 car，只检出 carNumber，
      则用 carNumber 作为“车辆存在代理证据”参与禁停侧边栏计时。
    - 车辆进入 noParking 后，如果仍在移动，只显示“经过禁停区/等待停止”，不累计违规时间。
    - 车辆在 noParking 内连续低速/静止后，才开始 3 秒计时；超过阈值触发告警。
    """
    now = time.time()
    cfg = STATE.config
    car_boxes, raw_car_count, plate_proxy_source_count, plate_proxy_count = _build_parking_candidates(boxes or [], cfg)
    zone_boxes = [dict(b) for b in (boxes or []) if is_no_parking_detection(b)]

    with STATE.lock:
        zones_mem = dict(getattr(STATE, "parking_zones", {}) or {})
        for z in zone_boxes:
            zid = _parking_zone_key(z)
            zones_mem[zid] = {
                "zone_id": zid,
                "bbox": [int(v) for v in (z.get("bbox") or [0, 0, 0, 0])],
                "class_id": int(z.get("class_id", 0) or 0),
                "label": str(z.get("class_display_name") or z.get("display_label") or "禁停区域"),
                "confidence": float(z.get("det_confidence", z.get("confidence", 0.0)) or 0.0),
                "last_seen_at": now,
                "frame_id": int(z.get("frame_id", 0) or 0),
            }
        zones_alive = {k: v for k, v in zones_mem.items() if now - float(v.get("last_seen_at", 0.0) or 0.0) <= float(cfg.parking_zone_memory_ttl)}
        STATE.parking_zones = zones_alive
        zones_for_match = list(zones_alive.values())

        parking_tracks = dict(getattr(STATE, "parking_tracks", {}) or {})
        seen_track_keys = set()
        for car in car_boxes:
            try:
                track_num = int(car.get("track_num", 0) or 0)
            except Exception:
                track_num = 0
            key = str(car.get("parking_track_key") or (f"C{track_num}" if track_num > 0 else car.get("track_id") or "unknown"))
            if not key or key == "unknown":
                continue
            seen_track_keys.add(key)
            car_bbox = [int(v) for v in (car.get("bbox") or [0, 0, 0, 0])]
            old = dict(parking_tracks.get(key) or {})

            raw_speed, speed_ema, stationary, stationary_since, last_moving_at = _parking_motion_state(
                old, car_bbox, now, frame_w, frame_h, cfg
            )

            best_zone = None
            best_score = 0.0
            best_reason = "outside"
            for zone in zones_for_match:
                inside, score, reason = _parking_relation_score(car_bbox, zone.get("bbox"), cfg)
                if inside and score > best_score:
                    best_zone, best_score, best_reason = zone, score, reason

            anchor = list(_parking_vehicle_anchor(car_bbox))
            common = {
                "track_num": track_num,
                "track_id": car.get("track_id") or (f"VEH-{track_num:03d}" if str(car.get("proxy_from", "car")) == "car" else f"PLATE-P{track_num:03d}"),
                "source_label": str(car.get("source_label") or ("车牌代理" if str(car.get("proxy_from", "")) == "plate_number" else "车辆框")),
                "proxy_from": str(car.get("proxy_from", "car")),
                "class_display_name": str(car.get("class_display_name") or car.get("display_label") or "车辆"),
                "associated_plate_track_id": str(car.get("associated_plate_track_id", "") or ""),
                "associated_plate_track_num": int(car.get("associated_plate_track_num", 0) or 0),
                "associated_plate_text": str(car.get("associated_plate_text", "") or ""),
                "associated_plate_bbox": car.get("associated_plate_bbox", []) or [],
                "bbox": car_bbox,
                "last_seen_at": now,
                "last_motion_ts": now,
                "last_anchor": anchor,
                "speed_raw": round(float(raw_speed), 5),
                "speed_ema": round(float(speed_ema), 5),
                "stationary": bool(stationary),
                "stationary_since": float(stationary_since or 0.0),
                "last_moving_at": float(last_moving_at or 0.0),
                "threshold_s": float(cfg.parking_violation_seconds),
                "frame_id": int(car.get("frame_id", 0) or 0),
            }

            if best_zone is not None:
                # 进入禁停区，但只有“停止确认”后才开始违规计时。
                old_zone_id = str(old.get("zone_id", ""))
                current_zone_id = str(best_zone.get("zone_id", ""))
                old_stopped_at = float(old.get("stopped_in_zone_at", 0.0) or 0.0)

                if stationary:
                    # 如果刚进入禁停区、刚从移动变静止、或换了禁停区，重新开始静止计时。
                    if not old_stopped_at or old_zone_id != current_zone_id or not bool(old.get("stationary", False)):
                        stopped_at = now
                    else:
                        stopped_at = old_stopped_at
                else:
                    stopped_at = 0.0

                dwell = max(0.0, now - stopped_at) if stopped_at else 0.0
                alert = bool(stopped_at and dwell >= float(cfg.parking_violation_seconds))
                event_id = old.get("event_id")
                if alert and not event_id:
                    event_id = f"PARK-{STATE.parking_event_seq:04d}"
                    STATE.parking_event_seq += 1

                status = "alert" if alert else ("counting" if stopped_at else ("moving_in_zone" if not stationary else "waiting_stable"))
                track_record = {
                    **common,
                    "zone_id": current_zone_id,
                    "zone_bbox": best_zone.get("bbox", []),
                    "zone_label": best_zone.get("label", "禁停区域"),
                    "in_zone": True,
                    "last_in_zone_at": now,
                    "stopped_in_zone_at": stopped_at,
                    "dwell_s": dwell,
                    "alert": alert,
                    "event_id": event_id,
                    "relation_score": round(float(best_score), 4),
                    "relation_reason": best_reason,
                    "parking_status": status,
                }
                if alert:
                    _append_parking_event_history(track_record)
                parking_tracks[key] = track_record
            elif old:
                # 离开禁停区或本帧没匹配到禁停区。给短暂 grace，避免单帧漏检导致状态闪烁。
                old.update(common)
                old["in_zone"] = False
                old["parking_status"] = "outside"
                if now - float(old.get("last_in_zone_at", 0.0) or 0.0) <= float(cfg.parking_exit_grace):
                    parking_tracks[key] = old
                else:
                    parking_tracks.pop(key, None)

        # 检测漏帧时保留短暂状态，但不会继续给“停止计时”增加 dwell，防止误报。
        for key, old in list(parking_tracks.items()):
            if key in seen_track_keys:
                continue
            if now - float(old.get("last_seen_at", 0.0) or 0.0) <= float(cfg.parking_exit_grace):
                old["parking_status"] = "lost_grace"
                parking_tracks[key] = old
            elif now - float(old.get("last_seen_at", 0.0) or 0.0) > float(cfg.parking_track_ttl):
                parking_tracks.pop(key, None)

        alive_tracks = {k: v for k, v in parking_tracks.items() if now - float(v.get("last_seen_at", 0.0) or 0.0) <= float(cfg.parking_track_ttl)}
        STATE.parking_tracks = alive_tracks

        active, alerts = [], []
        for tr in alive_tracks.values():
            item = {
                "track_num": int(tr.get("track_num", 0) or 0),
                "track_id": str(tr.get("track_id", "")),
                "bbox": tr.get("bbox", []),
                "zone_id": tr.get("zone_id", ""),
                "zone_bbox": tr.get("zone_bbox", []),
                "zone_label": tr.get("zone_label", "禁停区域"),
                "dwell_s": round(float(tr.get("dwell_s", 0.0) or 0.0), 2),
                "threshold_s": round(float(cfg.parking_violation_seconds), 2),
                "in_zone": bool(tr.get("in_zone", False)),
                "stationary": bool(tr.get("stationary", False)),
                "speed_norm": round(float(tr.get("speed_ema", 0.0) or 0.0), 5),
                "status": str(tr.get("parking_status", "outside")),
                "alert": bool(tr.get("alert", False)),
                "event_id": str(tr.get("event_id", "")),
                "relation_score": float(tr.get("relation_score", 0.0) or 0.0),
                "relation_reason": str(tr.get("relation_reason", "")),
                "source_label": str(tr.get("source_label", "车辆框")),
                "proxy_from": str(tr.get("proxy_from", "car")),
                "class_display_name": str(tr.get("class_display_name", "车辆")),
                "associated_plate_track_id": str(tr.get("associated_plate_track_id", "")),
                "associated_plate_track_num": int(tr.get("associated_plate_track_num", 0) or 0),
                "associated_plate_text": str(tr.get("associated_plate_text", "")),
                "vehicle_name": _parking_track_display_name(tr),
            }
            if item["in_zone"] or item["alert"] or item["status"] in {"lost_grace"}:
                active.append(item)
            if item["alert"]:
                alerts.append(item)
        STATE.status["parking_alerts"] = len(alerts)
        STATE.status["parking_active"] = len(active)
        STATE.status["parking_threshold_s"] = float(cfg.parking_violation_seconds)

    return {
        "enabled": True,
        "mode": "bytetrack_iou_stop_then_count_plate_proxy_v3",
        "threshold_s": float(cfg.parking_violation_seconds),
        "exit_grace_s": float(cfg.parking_exit_grace),
        "stationary_speed_norm": float(cfg.parking_stationary_speed_norm),
        "stationary_confirm_s": float(cfg.parking_stationary_confirm_seconds),
        "active": active,
        "alerts": alerts,
        "history": list(getattr(STATE, "parking_event_history", []) or [])[-30:][::-1],
        "zones": list(zones_for_match),
        "summary": {
            "car_count": int(raw_car_count),
            "plate_number_count": int(plate_proxy_source_count),
            "plate_proxy_count": int(plate_proxy_count),
            "candidate_count": int(len(car_boxes)),
            "zone_count": len(zones_for_match),
            "active_count": len(active),
            "alert_count": len(alerts),
        },
    }

def annotate_parking_to_boxes(boxes: List[Dict[str, Any]], parking_info: Dict[str, Any]) -> None:
    by_track = {int(p.get("track_num", 0) or 0): p for p in (parking_info.get("active") or [])}
    for box in boxes or []:
        try:
            tn = int(box.get("track_num", 0) or 0)
        except Exception:
            tn = 0
        p = by_track.get(tn)
        if p and is_heatmap_vehicle_detection(box):
            box["parking_in_zone"] = bool(p.get("in_zone"))
            box["parking_alert"] = bool(p.get("alert"))
            box["parking_dwell_s"] = float(p.get("dwell_s", 0.0) or 0.0)
            box["parking_threshold_s"] = float(p.get("threshold_s", 3.0) or 3.0)
            box["parking_zone_label"] = str(p.get("zone_label", "禁停区域"))
            box["parking_event_id"] = str(p.get("event_id", ""))
            box["parking_relation_score"] = float(p.get("relation_score", 0.0) or 0.0)
            box["parking_stationary"] = bool(p.get("stationary", False))
            box["parking_status"] = str(p.get("status", ""))
            box["parking_speed_norm"] = float(p.get("speed_norm", 0.0) or 0.0)
        else:
            box.setdefault("parking_in_zone", False)
            box.setdefault("parking_alert", False)
            box.setdefault("parking_dwell_s", 0.0)
            box.setdefault("parking_stationary", False)
            box.setdefault("parking_status", "")
            box.setdefault("parking_speed_norm", 0.0)


def load_models() -> None:
    model_info = active_model_info()
    STATE.update_status(
        backend="loading",
        detector_model=model_info["key"],
        active_model_name=model_info["name"],
        model_label=model_info["label"],
        model_display_label=model_info["display_label"],
        model_uses_ocr=bool(model_info["uses_ocr"]),
        message=f"正在预加载 {model_info['display_label']} ONNX 检测模型与轻量 OCR 识别模型。"
    )
    errors: List[str] = []
    # 切换模型前先清空旧 ONNX Session，避免 hearmap.onnx 缺失时继续误用旧的车牌模型。
    STATE.ort_session = None
    STATE.ort_input_name = ""
    STATE.ort_output_names = []
    STATE.ort_input_height = 0
    STATE.ort_input_width = 0
    STATE.ort_input_size = 0
    STATE.model_class_names = {}
    STATE.model_metadata = {}

    try:
        model_info = active_model_info()
        active_model_path = Path(model_info["file"])
        if ort is None:
            raise RuntimeError("未安装 onnxruntime / onnxruntime-gpu")
        if not active_model_path.exists():
            raise FileNotFoundError(f"未找到 {model_info['display_label']} ONNX 模型：{active_model_path}")

        preload_info = preload_cuda_runtime_for_onnx()
        log(f"ONNX CUDA DLL preload: ok={preload_info.get('ok')} message={preload_info.get('message')}")
        STATE.update_status(
            cuda_preload_ok=bool(preload_info.get("ok")),
            cuda_preload_message=str(preload_info.get("message", "")),
            cuda_dll_dirs=preload_info.get("dll_dirs", [])[:8],
        )

        available = ort.get_available_providers()
        cfg = STATE.config
        trt_cache_path = ROOT_DIR / str(cfg.trt_engine_cache_path)
        trt_cache_path.mkdir(parents=True, exist_ok=True)

        log(f"Loading ONNX model [{model_info['key']}]: {active_model_path}")
        log(f"ONNX available providers: {available}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = 1

        provider_errors: List[str] = []
        provider_attempts: List[List[str]] = []

        def provider_name(p: Any) -> str:
            return p[0] if isinstance(p, tuple) else str(p)

        # 逐级尝试，避免“环境有 CUDA/TensorRT，但 Session 静默落到 CPU”却不被发现。
        # TensorRT 尝试：TensorRT -> CUDA -> CPU；失败或实际首选落 CPU，则继续试 CUDA。
        plans: List[List[Any]] = []
        if cfg.enable_tensorrt and "TensorrtExecutionProvider" in available:
            plans.append([
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_engine_cache_enable": "1" if cfg.trt_engine_cache_enable else "0",
                        "trt_engine_cache_path": str(trt_cache_path),
                        "trt_fp16_enable": "1" if cfg.trt_fp16_enable else "0",
                    },
                ),
                "CUDAExecutionProvider" if "CUDAExecutionProvider" in available else "CPUExecutionProvider",
                "CPUExecutionProvider",
            ])
        if "CUDAExecutionProvider" in available:
            plans.append([
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": "0",
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": "1",
                    },
                ),
                "CPUExecutionProvider",
            ])
        plans.append(["CPUExecutionProvider"])

        selected_plan: List[Any] = []
        last_exc: Optional[Exception] = None
        for plan in plans:
            requested_names = [provider_name(x) for x in plan]
            provider_attempts.append(requested_names)
            log(f"ONNX provider attempt: {requested_names}")
            try:
                sess = ort.InferenceSession(str(active_model_path), sess_options=sess_options, providers=plan)
                actual = sess.get_providers()
                log(f"ONNX actual providers after attempt: {actual}")

                # 如果请求了 GPU，但实际 Session 首位仍是 CPU，视为该 GPU 计划失败，继续尝试下一档。
                requested_primary = requested_names[0] if requested_names else "CPUExecutionProvider"
                actual_primary = actual[0] if actual else "unknown"
                if requested_primary != "CPUExecutionProvider" and actual_primary == "CPUExecutionProvider":
                    provider_errors.append(f"{requested_primary} requested but actual={actual}")
                    continue

                STATE.ort_session = sess
                selected_plan = requested_names
                break
            except Exception as ep_exc:
                last_exc = ep_exc
                provider_errors.append(f"{requested_names}: {repr(ep_exc)}")
                log(f"ONNX provider init failed: {requested_names}: {ep_exc}")

        if STATE.ort_session is None:
            raise RuntimeError(f"ONNX Session 初始化失败: {last_exc}; errors={provider_errors}")

        STATE.ort_input_name = STATE.ort_session.get_inputs()[0].name
        STATE.ort_output_names = [o.name for o in STATE.ort_session.get_outputs()]
        input_h, input_w = resolve_onnx_input_hw(STATE.ort_session, int(cfg.imgsz))
        STATE.ort_input_height = int(input_h)
        STATE.ort_input_width = int(input_w)
        STATE.ort_input_size = int(max(input_h, input_w))
        log(f"ONNX input shape resolved: H={STATE.ort_input_height}, W={STATE.ort_input_width}, effective_imgsz={STATE.ort_input_size}")

        class_names, model_metadata = extract_onnx_class_names(STATE.ort_session, str(model_info["key"]))
        STATE.model_class_names = dict(class_names)
        STATE.model_metadata = dict(model_metadata)
        if class_names:
            log(f"ONNX class names resolved [{model_info['key']}]: {class_names}")
        else:
            log(f"ONNX class names not found in metadata [{model_info['key']}]; will display class_id fallback labels.")

        # Warmup 的作用：让 CUDA/TensorRT 把首轮初始化、kernel 选择、engine 构建提前完成，
        # 后续正式检测时的耗时数据才更真实。
        warm_t0 = time.perf_counter()
        try:
            dummy = np.zeros((1, 3, int(STATE.ort_input_height or cfg.imgsz), int(STATE.ort_input_width or cfg.imgsz)), dtype=np.float32)
            for _ in range(max(0, int(cfg.warmup_runs))):
                STATE.ort_session.run(STATE.ort_output_names or None, {STATE.ort_input_name: dummy})
        except Exception as warm_exc:
            log(f"ONNX warmup skipped/failed: {warm_exc}")
            provider_errors.append(f"warmup: {repr(warm_exc)}")
        warm_ms = (time.perf_counter() - warm_t0) * 1000.0

        actual_providers = STATE.ort_session.get_providers()
        used_provider = actual_providers[0] if actual_providers else "unknown"
        STATE.update_status(
            yolo_ready=True,
            yolo_provider=used_provider,
            actual_providers=actual_providers,
            available_providers=list(available),
            session_provider_requested=selected_plan,
            provider_init_errors=provider_errors[-5:],
            trt_cache_path=str(trt_cache_path),
            trt_warmup_ms=round(warm_ms, 2),
            model_path=str(active_model_path),
            model_input_width=int(STATE.ort_input_width or 0),
            model_input_height=int(STATE.ort_input_height or 0),
            model_input_size=int(STATE.ort_input_size or cfg.imgsz),
            config_imgsz=int(cfg.imgsz),
            detector_model=model_info["key"],
            active_model_name=model_info["name"],
            model_label=model_info["label"],
            model_display_label=model_info["display_label"],
            model_uses_ocr=bool(model_info["uses_ocr"]),
            model_class_names={str(k): v for k, v in STATE.model_class_names.items()},
            model_metadata=STATE.model_metadata,
        )
    except Exception as exc:
        errors.append(f"ONNX加载失败: {exc}")
        log(traceback.format_exc())
        STATE.update_status(yolo_ready=False)

    # OCR 模型与四个检测 ONNX 无关。首次启动加载一次，后续切换检测模型时直接复用，
    # 避免每次切换 best/hearmap/stop/normal 都重新初始化 PaddleOCR。
    if STATE.ocr_model is None:
        try:
            from paddleocr import TextRecognition
            model_name = STATE.config.ocr_model_name
            log(f"Loading PaddleOCR TextRecognition model: {model_name}")
            try:
                STATE.ocr_model = TextRecognition(model_name=model_name)
            except Exception:
                log("Tiny OCR model unavailable, fallback to default TextRecognition().")
                STATE.ocr_model = TextRecognition()
                model_name = "default"
            STATE.update_status(ocr_ready=True, ocr_model=model_name)
        except Exception as exc:
            errors.append(f"OCR加载失败: {exc}")
            log(traceback.format_exc())
            STATE.update_status(ocr_ready=False)
    else:
        STATE.update_status(ocr_ready=True, ocr_model=STATE.config.ocr_model_name)

    ready = bool(STATE.ort_session is not None and STATE.ocr_model is not None)
    STATE.update_status(
        backend="ready" if ready else "degraded",
        models_ready=ready,
        message=f"{active_model_info()['display_label']} ONNX 与 OCR 模型预加载完成。" if ready else "；".join(errors),
    )


def letterbox(im: np.ndarray, new_shape: int = 416, color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    shape = im.shape[:2]  # h, w
    r = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape - new_unpad[0]
    dh = new_shape - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def preprocess_for_onnx(frame: np.ndarray, imgsz: int) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    img, ratio, dwdh = letterbox(frame, imgsz)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img[None, :, :, :], ratio, dwdh


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(1.0, (x2 - x1) * (y2 - y1))
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def parse_yolo_output(output: np.ndarray, frame_shape: Tuple[int, int], ratio: float, dwdh: Tuple[float, float], imgsz: int, conf_thres: float, iou_thres: float, label: str = "plate", model_info: Optional[Dict[str, Any]] = None, class_names: Optional[Dict[int, str]] = None) -> List[Dict[str, Any]]:
    pred = np.asarray(output)
    if pred.ndim == 3:
        pred = pred[0]
    # Ultralytics YOLOv8 ONNX 常见输出为 [5, 3549] / [84, 8400]，需要转为 [N, C]
    if pred.ndim == 2 and pred.shape[0] < pred.shape[1] and pred.shape[0] <= 128:
        pred = pred.T
    if pred.ndim != 2 or pred.shape[1] < 5:
        return []

    h0, w0 = frame_shape
    dw, dh = dwdh
    boxes_list: List[List[float]] = []
    scores_list: List[float] = []
    cls_list: List[int] = []

    for row in pred:
        if row.shape[0] < 5:
            continue
        xywh = row[:4].astype(np.float32)
        # 如果导出模型输出归一化坐标，则转为输入尺寸坐标。
        if float(np.max(np.abs(xywh))) <= 2.0:
            xywh = xywh * float(imgsz)

        if row.shape[0] == 5:
            score = float(row[4])
            cls_id = 0
        else:
            cls_scores = row[4:]
            cls_id = int(np.argmax(cls_scores))
            score = float(cls_scores[cls_id])
            # 兼容 YOLOv5 风格 [x,y,w,h,obj,cls...]，如果 cls 很小则尝试 obj*cls。
            if row.shape[0] >= 6 and float(row[4]) <= 1.0 and float(np.max(row[5:])) <= 1.0:
                score_alt = float(row[4]) * float(np.max(row[5:]))
                if score_alt > score * 0.65:
                    score = max(score, score_alt)
                    cls_id = int(np.argmax(row[5:]))

        if score < conf_thres:
            continue

        cx, cy, bw, bh = [float(v) for v in xywh]
        x1 = (cx - bw / 2.0 - dw) / ratio
        y1 = (cy - bh / 2.0 - dh) / ratio
        x2 = (cx + bw / 2.0 - dw) / ratio
        y2 = (cy + bh / 2.0 - dh) / ratio
        x1 = max(0.0, min(float(w0 - 1), x1))
        y1 = max(0.0, min(float(h0 - 1), y1))
        x2 = max(0.0, min(float(w0 - 1), x2))
        y2 = max(0.0, min(float(h0 - 1), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) < 4 or (y2 - y1) < 4:
            continue
        boxes_list.append([x1, y1, x2, y2])
        scores_list.append(score)
        cls_list.append(cls_id)

    if not boxes_list:
        return []
    boxes = np.asarray(boxes_list, dtype=np.float32)
    scores = np.asarray(scores_list, dtype=np.float32)
    keep = nms_numpy(boxes, scores, iou_thres)
    detections: List[Dict[str, Any]] = []
    active_info = model_info or {"key": "", "display_label": label}
    for idx in keep:
        x1, y1, x2, y2 = boxes[idx].tolist()
        cls_id = int(cls_list[idx])
        cls_name = str((class_names or {}).get(cls_id, ""))
        cls_display = class_display_name(active_info, cls_id, class_names)
        model_key = str(active_info.get("key", ""))
        sem_type = semantic_class_type(model_key, cls_id, cls_name or cls_display)
        unified_name = unified_display_name(model_key, cls_id, cls_name or cls_display, cls_display)
        detections.append({
            "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
            "confidence": round(float(scores[idx]), 4),
            "class_id": cls_id,
            "label": label,
            "class_name": cls_name,
            "raw_class_name": cls_name or cls_display,
            "class_display_name": unified_name,
            "display_label": unified_name,
            "semantic_type": sem_type,
            "heatmap_eligible": sem_type == "vehicle",
        })
    return detections


def detect_plates_onnx(frame: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
    cfg = STATE.config
    t0 = time.perf_counter()

    # 与动态模型切换共用 inference_lock：切换时只等待当前这一轮推理结束，
    # 不停止 capture_loop，也不重连手机视频源。
    with STATE.inference_lock:
        session = STATE.ort_session
        input_name = STATE.ort_input_name
        output_names = list(STATE.ort_output_names)
        class_names = dict(STATE.model_class_names)
        if session is None or not input_name:
            return [], 0.0

        effective_imgsz = current_model_input_size()
        inp, ratio, dwdh = preprocess_for_onnx(frame, effective_imgsz)
        t1 = time.perf_counter()
        outputs = session.run(output_names or None, {input_name: inp})
        t2 = time.perf_counter()

    model_info = active_model_info(cfg.detector_model)
    detections = parse_yolo_output(
        outputs[0],
        frame.shape[:2],
        ratio,
        dwdh,
        effective_imgsz,
        cfg.conf,
        cfg.nms_iou,
        label=str(model_info["label"]),
        model_info=model_info,
        class_names=class_names,
    )
    t3 = time.perf_counter()
    pre_ms = (t1 - t0) * 1000.0
    infer_ms = (t2 - t1) * 1000.0
    post_ms = (t3 - t2) * 1000.0
    total_ms = (t3 - t0) * 1000.0
    perf_add("yolo_pre_ms", pre_ms)
    perf_add("yolo_infer_ms", infer_ms)
    perf_add("yolo_post_ms", post_ms)
    perf_add("yolo_total_ms", total_ms)
    STATE.update_status(
        yolo_pre_ms=round(pre_ms, 2),
        yolo_infer_ms=round(infer_ms, 2),
        yolo_post_ms=round(post_ms, 2),
        model_input_size=int(effective_imgsz),
    )
    return detections, total_ms


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aa = max(1, (ax2 - ax1) * (ay2 - ay1))
    ba = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(aa + ba - inter + 1e-6)


def current_source_type() -> str:
    with STATE.lock:
        return str(STATE.status.get("source_type", "idle"))


def track_ttl_for_current_source() -> float:
    cfg = STATE.config
    return cfg.camera_track_ttl if current_source_type() == "camera" else cfg.file_track_ttl


def display_ttl_for_current_source() -> float:
    cfg = STATE.config
    # YOLO 低频时给一点缓冲，避免正常情况下闪烁；camera 模式仍明显短于旧版 2 秒。
    base = cfg.camera_display_ttl if current_source_type() == "camera" else cfg.file_display_ttl
    return max(base, float(cfg.yolo_interval) * 2.8)


def max_frame_lag_for_current_source() -> int:
    cfg = STATE.config
    return int(cfg.camera_max_frame_lag if current_source_type() == "camera" else cfg.file_max_frame_lag)


def ocr_result_max_age_for_current_source() -> float:
    cfg = STATE.config
    return cfg.camera_ocr_result_max_age if current_source_type() == "camera" else cfg.file_ocr_result_max_age


def clear_ocr_queue() -> None:
    try:
        while True:
            STATE.ocr_queue.get_nowait()
    except Exception:
        pass


def reset_visual_state(reason: str = "") -> None:
    """只清理视觉状态，不停止视频流。用于摄像头晃动/切换位置后清理旧框与旧 OCR。"""
    with STATE.lock:
        STATE.tracks.clear()
        STATE.latest_yolo_boxes = []
        STATE.latest_detections = []
        STATE.latest_result = {
            "plates": [],
            "detections": [],
            "tracks": [],
            "parking_monitor": {"active": [], "alerts": [], "zones": []},
            "parking_alert_count": 0,
            "message": reason or "检测状态已重置。",
            "timestamp": time.time(),
            "frame_id": STATE.latest_frame_id,
        }
        STATE.parking_tracks.clear()
        STATE.parking_zones.clear()
        STATE.status["parking_alerts"] = 0
        STATE.status["parking_active"] = 0
    clear_ocr_queue()


def detect_global_motion(frame: np.ndarray) -> float:
    """估计整幅画面的全局变化。摄像头晃动时分数会明显高于局部车辆运动。"""
    if frame is None or frame.size == 0:
        return 0.0
    try:
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        with STATE.lock:
            prev = STATE.motion_gray
            STATE.motion_gray = gray
        if prev is None or prev.shape != gray.shape:
            return 0.0
        return float(cv2.absdiff(gray, prev).mean())
    except Exception:
        return 0.0


def match_track(bbox: Tuple[int, int, int, int], det_conf: float, frame_id: int, frame_ts: float, frame_shape: Optional[Tuple[int, int]] = None, detector_model: str = "plate", semantic_type: str = "plate_number", class_id: int = 0) -> PlateTrack:
    """短期车牌跟踪。

    这是后台不可见的“短时车牌关联”，不是面向用户展示的车辆跟踪功能。
    它只承担两个内部职责：
    1. 临时关联连续帧中的同一块车牌；
    2. 合并多次 OCR 结果完成投票纠错。

    车牌识别界面不会显示 track_id、跟踪编号或车辆跟踪卡片。

    对手持摄像头做了更宽容的中心点/尺寸匹配：摄像头移动时，同一块车牌在画面中可能整体跳动，
    不能只依赖 IoU，否则 track_id 会频繁断裂，OCR 文字无法复用。
    """
    now = time.time()
    ttl = track_ttl_for_current_source()
    h, w = frame_shape if frame_shape else (720, 1280)
    frame_diag = max(1.0, (float(w) ** 2 + float(h) ** 2) ** 0.5)

    bx1, by1, bx2, by2 = bbox
    bw = max(1.0, float(bx2 - bx1))
    bh = max(1.0, float(by2 - by1))
    barea = bw * bh
    cx = (bx1 + bx2) / 2.0
    cy = (by1 + by2) / 2.0

    with STATE.lock:
        STATE.tracks = [t for t in STATE.tracks if t.is_recent(now, ttl=ttl)]
        best: Optional[PlateTrack] = None
        best_score = -1.0

        for tr in STATE.tracks:
            # 同一 YOLO 帧中的两个检测框不能占用同一个 track。
            # 否则相邻两块车牌会共享 OCR 候选，造成文本串牌。
            if int(getattr(tr, "last_frame_id", 0) or 0) == int(frame_id):
                continue

            # 不同语义目标不能共用同一个 track_id。
            # 例如 car 框内含 carNumber，如果不区分语义，会出现二者都显示 #2 的混乱问题。
            if str(getattr(tr, "semantic_type", "")) != str(semantic_type):
                continue
            tx1, ty1, tx2, ty2 = tr.bbox
            tw = max(1.0, float(tx2 - tx1))
            th = max(1.0, float(ty2 - ty1))
            tarea = tw * th
            tcx, tcy = tr.center()
            dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
            box_iou = iou(tr.bbox, bbox)

            # 小范围短时跟踪：只有位置没有发生明显跳跃、且框尺寸大致相近时才视为同一车牌。
            track_age = max(0.0, now - float(tr.last_seen or now))
            if current_source_type() == "camera":
                max_jump_ratio = 0.115 + min(0.035, track_age * 0.04)
            else:
                max_jump_ratio = 0.080 + min(0.025, track_age * 0.025)

            # 大跳变且几乎没有重叠，直接拒绝，防止相邻车牌共用投票历史。
            if box_iou < 0.04 and dist > frame_diag * max_jump_ratio:
                continue

            dist_score = max(0.0, 1.0 - dist / max(1.0, frame_diag * max_jump_ratio))
            size_score = max(0.0, 1.0 - abs(barea - tarea) / max(barea, tarea, 1.0))
            if size_score < 0.28 and box_iou < 0.10:
                continue

            score = 0.56 * box_iou + 0.31 * dist_score + 0.13 * size_score
            if box_iou >= 0.25:
                score += 0.12
            if score > best_score:
                best_score = score
                best = tr

        min_score = 0.26 if current_source_type() == "camera" else 0.32
        if best is not None and best_score >= min_score:
            best.bbox = bbox
            best.det_conf = det_conf
            best.last_seen = now
            best.last_frame_id = frame_id
            best.last_frame_ts = frame_ts
            best.detector_model = str(detector_model)
            best.semantic_type = str(semantic_type)
            best.class_id = int(class_id)
            return best

        new_track = PlateTrack(
            track_id=STATE.next_track_id,
            bbox=bbox,
            det_conf=det_conf,
            last_seen=now,
            last_frame_id=frame_id,
            last_frame_ts=frame_ts,
            detector_model=str(detector_model),
            semantic_type=str(semantic_type),
            class_id=int(class_id),
        )
        STATE.next_track_id += 1
        STATE.tracks.append(new_track)
        return new_track

def safe_crop(frame: np.ndarray, bbox: Tuple[int, int, int, int], pad_ratio: float = 0.10) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw <= 2 or bh <= 2:
        return None
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)
    crop = frame[y1:y2, x1:x2]
    return crop.copy() if crop.size > 0 else None


def preprocess_plate(img: np.ndarray) -> Optional[np.ndarray]:
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return None
    # 车牌过小容易丢省份字，适当放大。
    target_w = 180
    if w < target_w:
        scale = target_w / max(w, 1)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # 轻量增强，不做强二值化。
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    except Exception:
        pass
    return img


def extract_text_recognition_result(output: Any) -> Tuple[str, float]:
    if not output:
        return "", 0.0
    res_obj = output[0]
    data = getattr(res_obj, "json", None)
    if callable(data):
        data = data()
    if isinstance(data, dict):
        inner = data.get("res", data)
        if "rec_text" in inner:
            return str(inner.get("rec_text", "")), float(inner.get("rec_score", 0.0) or 0.0)
        if "rec_texts" in inner:
            texts = inner.get("rec_texts", [])
            scores = inner.get("rec_scores", [])
            text = "".join([str(t) for t in texts if t])
            score = float(max(scores)) if scores else 0.0
            return text, score
    return "", 0.0


def recognize_plate(crop: np.ndarray) -> Tuple[str, float, float]:
    if STATE.ocr_model is None:
        return "", 0.0, 0.0
    crop = preprocess_plate(crop)
    if crop is None:
        return "", 0.0, 0.0
    t0 = time.perf_counter()
    output = STATE.ocr_model.predict(input=crop, batch_size=1)
    ocr_ms = (time.perf_counter() - t0) * 1000.0
    perf_add("ocr_recognize_ms", ocr_ms)
    raw, conf = extract_text_recognition_result(output)
    return PlateVoting.clean(raw), float(conf or 0.0), ocr_ms


def should_schedule_ocr(track: PlateTrack, now: float) -> bool:
    cfg = STATE.config
    if track.ocr_pending:
        return False

    elapsed = now - track.last_ocr_time

    # 未稳定：前几次快速连识别，超过突发次数仍不稳定时自动降频，
    # 避免模糊车牌永远以 4 次/秒占用 OCR。
    if not track.stable_text:
        burst_limit = max(1, int(getattr(cfg, "ocr_burst_max_attempts", 6)))
        interval = (
            float(cfg.ocr_min_interval)
            if track.ocr_attempts < burst_limit
            else float(getattr(cfg, "ocr_unstable_interval", 0.60))
        )
        return elapsed >= max(0.05, interval)

    # 真正稳定后只做低频复核，不再为了填满 max_vote_history 持续高频识别。
    return elapsed >= max(float(cfg.stable_recheck_interval), float(cfg.ocr_min_interval))


def schedule_ocr(track: PlateTrack, frame: np.ndarray, frame_id: int, frame_ts: float) -> None:
    crop = safe_crop(frame, track.bbox)
    if crop is None:
        return
    with STATE.lock:
        track.ocr_pending = True
        track.last_ocr_time = time.time()
        track.ocr_attempts += 1
    item = {
        "track_id": track.track_id,
        "crop": crop,
        "timestamp": time.time(),
        "frame_id": int(frame_id),
        "frame_ts": float(frame_ts or 0.0),
        "bbox": tuple(int(v) for v in track.bbox),
    }
    try:
        STATE.ocr_queue.put_nowait(item)
        perf_inc("ocr_scheduled")
    except queue.Full:
        perf_inc("ocr_queue_drop")
        dropped_item = None
        try:
            dropped_item = STATE.ocr_queue.get_nowait()
        except Exception:
            dropped_item = None

        # 被挤掉的任务永远不会进入 ocr_loop，必须立即释放它对应 track 的 pending。
        # 否则该车牌会永久停在“识别中”，后续再也不会提交 OCR。
        if isinstance(dropped_item, dict):
            dropped_track_id = int(dropped_item.get("track_id", -1) or -1)
            with STATE.lock:
                dropped_track = next(
                    (t for t in STATE.tracks if t.track_id == dropped_track_id),
                    None,
                )
                if dropped_track is not None:
                    dropped_track.ocr_pending = False

        try:
            STATE.ocr_queue.put_nowait(item)
            perf_inc("ocr_scheduled")
        except Exception:
            with STATE.lock:
                track.ocr_pending = False


def add_ocr_candidate(track_id: int, text: str, conf: float, source_frame_id: int = 0, source_frame_ts: float = 0.0) -> None:
    with STATE.lock:
        track = next((t for t in STATE.tracks if t.track_id == track_id), None)
        if not track:
            return
        track.ocr_pending = False
        # OCR 是异步返回的：如果对应 track 已经很久没有被当前画面检测到，直接丢弃，避免旧文字贴到新画面。
        now = time.time()
        if not track.is_recent(now, ttl=track_ttl_for_current_source()):
            return
        if source_frame_id and STATE.latest_frame_id - int(source_frame_id) > max_frame_lag_for_current_source() * 2:
            return
        if conf < STATE.config.min_ocr_conf or not text:
            return
        candidate = PlateVoting.make_candidate(text, conf, source_frame_id=source_frame_id, source_frame_ts=source_frame_ts)
        if not candidate:
            return
        track.candidates.append(candidate)
        track.candidates = track.candidates[-STATE.config.max_vote_history:]
        stable_text, stable_score, _debug = PlateVoting.vote(track, STATE.config.min_stable_votes)
        if stable_text:
            # 如果新分数更好，或原本没有稳定结果，就更新。
            if not track.stable_text or stable_score >= track.stable_score * 0.92:
                track.stable_text = stable_text
                track.stable_score = stable_score
                track.stable_at = time.time()


def ocr_loop() -> None:
    log("OCR worker started.")
    last_time = time.time()
    count = 0
    while not STATE.stop_event.is_set():
        try:
            item = STATE.ocr_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        track_id = int(item.get("track_id", -1))
        crop = item.get("crop")
        item_ts = float(item.get("timestamp", time.time()))
        item_frame_id = int(item.get("frame_id", 0) or 0)
        item_frame_ts = float(item.get("frame_ts", 0.0) or 0.0)
        try:
            text, conf, ocr_ms = recognize_plate(crop)
            now_done = time.time()
            with STATE.lock:
                latest_id = int(STATE.latest_frame_id)
            # OCR 太旧时只释放 pending，不进入投票，避免手持晃动时文字严重滞后。
            too_old_time = now_done - item_ts > ocr_result_max_age_for_current_source()
            too_old_frame = item_frame_id and latest_id - item_frame_id > max_frame_lag_for_current_source() * 2
            if too_old_time or too_old_frame:
                with STATE.lock:
                    tr = next((t for t in STATE.tracks if t.track_id == track_id), None)
                    if tr:
                        tr.ocr_pending = False
                perf_inc("ocr_expired_drop")
                STATE.update_status(message="已丢弃过期 OCR 结果，避免旧文字贴到当前画面。")
                continue
            add_ocr_candidate(track_id, text, conf, item_frame_id, item_frame_ts)
            perf_inc("ocr_done")
            count += 1
            now = time.time()
            if now - last_time >= 1.0:
                STATE.update_status(ocr_fps=round(count / max(1e-6, now - last_time), 2))
                count = 0
                last_time = now
            STATE.update_status(ocr_ms=round(ocr_ms, 2))
        except Exception as exc:
            log(f"OCR error: {exc}")
            log(traceback.format_exc())
            with STATE.lock:
                tr = next((t for t in STATE.tracks if t.track_id == track_id), None)
                if tr:
                    tr.ocr_pending = False


def capture_loop(source: str, source_type: str) -> None:
    STATE.update_status(running=True, source_type=source_type, source=source, message="正在打开视频源...", capture_fps=0.0, frame_id=0)
    cap = cv2.VideoCapture(source)
    if source_type == "camera":
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        STATE.update_status(running=False, message=f"视频源打开失败：{source}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if video_fps <= 1 or video_fps > 60:
        video_fps = 25.0
    frame_delay = 1.0 / video_fps if source_type == "file" else 0.0

    STATE.update_status(message="视频源已连接：显示线程与 AI 推理线程已解耦。")
    frames = 0
    last_fps_t = time.time()
    fid = 0

    while not STATE.stop_event.is_set():
        loop_t = time.time()
        read_t0 = time.perf_counter()
        ret, frame = cap.read()
        perf_add("capture_read_ms", (time.perf_counter() - read_t0) * 1000.0)
        if not ret or frame is None:
            if source_type == "file":
                STATE.source_finished = True
                STATE.update_status(message="本地视频播放完成。")
                break
            # 手机流断开时尝试轻量重连。
            STATE.update_status(message="视频流暂时中断，正在尝试重连...")
            cap.release()
            time.sleep(0.8)
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        fid += 1
        frame_ts_now = time.time()
        with STATE.lock:
            STATE.latest_frame = frame
            STATE.latest_frame_id = fid
            STATE.latest_frame_ts = frame_ts_now
            STATE.status["frame_id"] = fid
            STATE.status["frame_width"] = int(frame.shape[1])
            STATE.status["frame_height"] = int(frame.shape[0])

        # 手持/移动摄像头防拖影：检测到全局画面剧烈变化时，清空旧 track 和 OCR 队列。
        if source_type == "camera" and STATE.config.motion_reset_enabled:
            motion_score = detect_global_motion(frame)
            with STATE.lock:
                STATE.motion_score = motion_score
                STATE.status["motion_score"] = round(motion_score, 2)
            now_motion = time.time()
            if motion_score >= STATE.config.motion_reset_score and now_motion - STATE.last_motion_reset_t >= STATE.config.motion_reset_min_interval:
                STATE.last_motion_reset_t = now_motion
                reset_visual_state(f"检测到摄像头晃动，已清空旧框/旧OCR：motion={motion_score:.1f}")
        frames += 1
        now = time.time()
        if now - last_fps_t >= 1.0:
            STATE.update_status(capture_fps=round(frames / max(1e-6, now - last_fps_t), 2))
            frames = 0
            last_fps_t = now
        if frame_delay > 0:
            spent = time.time() - loop_t
            if spent < frame_delay:
                time.sleep(frame_delay - spent)

    cap.release()


def yolo_loop() -> None:
    """YOLO 实时检测线程。

    关键重构：
    - YOLO 当前帧检测到的 bbox 立即进入 STATE.latest_yolo_boxes，用于前端画框；
    - OCR 只异步更新 track 上的 stable_text / candidates，不再决定是否画框；
    - latest_result 也以“当前 YOLO 框”为主，而不是以历史 active_tracks 为主。
    """
    log("YOLO ONNX worker started.")
    last_processed_id = -1
    last_infer_t = 0.0
    count = 0
    fps_t = time.time()

    while not STATE.stop_event.is_set():
        # 切换 ONNX Session 时只暂停推理；capture_loop 与 MJPEG 输出继续运行。
        if STATE.model_switch_event.is_set():
            time.sleep(0.01)
            continue
        cfg = STATE.config
        model_generation = int(STATE.model_generation)
        model_info = active_model_info(cfg.detector_model)
        detector_key = model_info["key"]
        is_plate_model = bool(model_info["uses_ocr"])
        track_prefix = "PLATE" if is_plate_model else ("STOP" if detector_key == "stop" else ("NOR" if detector_key == "normal" else "VEH"))
        now = time.time()
        if now - last_infer_t < cfg.yolo_interval:
            time.sleep(0.006)
            continue

        snapshot_t0 = time.perf_counter()
        with STATE.lock:
            frame = None if STATE.latest_frame is None else STATE.latest_frame.copy()
            fid = STATE.latest_frame_id
            frame_ts = STATE.latest_frame_ts
        perf_add("frame_snapshot_ms", (time.perf_counter() - snapshot_t0) * 1000.0)
        if frame is None or fid == last_processed_id:
            time.sleep(0.01)
            continue

        last_processed_id = fid
        last_infer_t = now
        try:
            detections, yolo_ms = detect_plates_onnx(frame)
            # 若当前帧推理期间发生了模型切换，丢弃这次旧模型结果，避免旧框闪回。
            if model_generation != int(STATE.model_generation):
                continue
            detect_time = time.time()
            current_boxes_out: List[Dict[str, Any]] = []
            plates_out: List[Dict[str, Any]] = []
            tracks_out: List[Dict[str, Any]] = []
            h, w = frame.shape[:2]

            # 新版车道线逻辑只属于 normal.onnx 模式：
            # 不启动独立线程，不在 plate/vehicle/stop 三种模式中执行。
            if detector_key == "normal":
                try:
                    normal_lane_info = STATE.normal_lane_detector.process(frame, fid)
                except Exception as lane_exc:
                    normal_lane_info = STATE.normal_lane_detector.empty_result(
                        frame_id=fid,
                        frame_width=w,
                        frame_height=h,
                        status="error",
                        message=f"正常道路车道线检测失败：{lane_exc}",
                    )
                    log(f"Normal lane detection error: {lane_exc}")
                STATE.normal_lane_result = normal_lane_info
                perf_add("normal_lane_ms", float(normal_lane_info.get("processing_ms", 0.0) or 0.0))
            else:
                normal_lane_info = STATE.normal_lane_detector.empty_result(
                    frame_id=fid,
                    frame_width=w,
                    frame_height=h,
                    status="disabled",
                    message="当前不是 normal.onnx 模式，车道线算法未运行。",
                )

            normal_roi_pixels = normal_lane_info.get("roi", []) if detector_key == "normal" else []
            track_t0 = time.perf_counter()

            for det in detections:
                bbox = tuple(int(v) for v in det["bbox"])
                det_conf = float(det.get("confidence", 0.0))
                class_id = int(det.get("class_id", 0) or 0)
                raw_class = str(det.get("class_name") or det.get("raw_class_name") or det.get("class_display_name") or "")
                sem_type = semantic_class_type(detector_key, class_id, raw_class)
                det["semantic_type"] = sem_type
                det["heatmap_eligible"] = sem_type == "vehicle"
                det["class_display_name"] = unified_display_name(detector_key, class_id, raw_class, str(det.get("class_display_name") or model_info["display_label"]))
                det["display_label"] = det["class_display_name"]
                prefix_by_sem = {"vehicle": "VEH", "plate_number": "PLATE", "no_parking": "STOP", "normal_zone": "NOR"}
                det_track_prefix = prefix_by_sem.get(sem_type, track_prefix)
                track = match_track(bbox, det_conf, fid, frame_ts, frame_shape=(h, w), detector_model=detector_key, semantic_type=sem_type, class_id=class_id)

                use_ocr_for_this = bool(is_plate_model or sem_type == "plate_number")
                if use_ocr_for_this and should_schedule_ocr(track, detect_time):
                    schedule_ocr(track, frame, fid, frame_ts)

                # 画框数据来自当前 YOLO 检测框；文字来自 track 缓存。
                # carNumber/车牌区域和独立车牌模型统一：可复用 OCR 文本。
                # car 才进入右侧热力图；noParking/normal/carNumber 不进入热力热点。
                if use_ocr_for_this:
                    stable = bool(track.stable_text and len(track.candidates) >= STATE.config.min_stable_votes)
                    preview_text, preview_score = PlateVoting.preview(track)
                    cached_text = track.stable_text or preview_text
                    cached_score = track.stable_score or preview_score
                    ocr_status = (
                        "done"
                        if stable
                        else ("ocr_pending" if track.ocr_pending else ("candidate" if cached_text else "waiting_ocr"))
                    )
                    display_label = cached_text or str(det.get("class_display_name") or "车牌区域")
                else:
                    stable = False
                    cached_text = ""
                    cached_score = 0.0
                    ocr_status = "not_used"
                    display_label = str(det.get("class_display_name") or det.get("class_name") or model_info["display_label"])

                road_assignment = assign_bbox_to_road(bbox, w, h) if sem_type == "vehicle" else {}
                box_item = {
                    "track_num": int(track.track_id),
                    "track_id": f"{det_track_prefix}-{track.track_id:03d}",
                    "bbox": list(bbox),                       # 当前 YOLO 框，显示层只画这个
                    "det_confidence": round(det_conf, 4),
                    "class_id": int(det.get("class_id", 0)),
                    "class_name": str(det.get("class_name", "") or ""),
                    "class_display_name": str(det.get("class_display_name", display_label) or display_label),
                    "semantic_type": sem_type,
                    "heatmap_eligible": sem_type == "vehicle",
                    "plate_text": cached_text,                 # OCR 缓存文本，可为空
                    "stable_text": track.stable_text,
                    "ocr_confidence": round(float(cached_score), 4),
                    "stable_score": round(float(track.stable_score), 4),
                    "stable": stable,
                    "votes": len(track.candidates),
                    "ocr_pending": bool(track.ocr_pending),
                    "ocr_status": ocr_status,
                    "frame_id": int(fid),
                    "frame_ts": float(frame_ts or 0.0),
                    "detected_at": detect_time,
                    "age_ms": 0.0,
                    "label": display_label,
                    "class_id": int(det.get("class_id", 0)),
                    "class_name": str(det.get("class_name", "") or ""),
                    "class_display_name": str(det.get("class_display_name", display_label) or display_label),
                    "semantic_type": sem_type,
                    "heatmap_eligible": sem_type == "vehicle",
                    "detector_model": detector_key,
                    "model_label": str(model_info["label"]),
                    "model_display_label": str(model_info["display_label"]),
                    "road_assignment": road_assignment,
                    "road_id": road_assignment.get("road_id", "") if road_assignment else "",
                    "road_name": road_assignment.get("road_name", "") if road_assignment else "",
                    "road_segment": road_assignment.get("segment", "") if road_assignment else "",
                }

                if detector_key == "normal":
                    center_point = (
                        (float(bbox[0]) + float(bbox[2])) / 2.0,
                        (float(bbox[1]) + float(bbox[3])) / 2.0,
                    )
                    inside_normal_roi = point_inside_polygon(center_point, normal_roi_pixels)
                    if inside_normal_roi and sem_type == "vehicle":
                        normal_road_status = "normal_vehicle"
                        box_item["class_display_name"] = "正常车辆"
                        box_item["display_label"] = "正常车辆"
                        box_item["label"] = "正常车辆"
                    elif inside_normal_roi:
                        normal_road_status = "inside_roi_unclassified"
                    else:
                        normal_road_status = "outside_normal_roi"
                    box_item["inside_normal_roi"] = bool(inside_normal_roi)
                    box_item["normal_road_status"] = normal_road_status
                    det["inside_normal_roi"] = bool(inside_normal_roi)
                    det["normal_road_status"] = normal_road_status

                current_boxes_out.append(box_item)
                plates_out.append(box_item.copy())
                tracks_out.append({
                    "track_id": f"{det_track_prefix}-{track.track_id:03d}",
                    "track_num": int(track.track_id),
                    "bbox": list(bbox),
                    "frame_id": int(fid),
                    "age_ms": 0.0,
                    "label": display_label,
                    "class_id": int(det.get("class_id", 0)),
                    "class_name": str(det.get("class_name", "") or ""),
                    "class_display_name": str(det.get("class_display_name", display_label) or display_label),
                    "semantic_type": sem_type,
                    "heatmap_eligible": sem_type == "vehicle",
                    "detector_model": detector_key,
                    "stable": stable,
                    "votes": len(track.candidates),
                    "ocr_status": ocr_status,
                })
                if detector_key == "normal" and tracks_out:
                    tracks_out[-1]["inside_normal_roi"] = bool(box_item.get("inside_normal_roi", False))
                    tracks_out[-1]["normal_road_status"] = str(box_item.get("normal_road_status", ""))
                    if box_item.get("normal_road_status") == "normal_vehicle":
                        tracks_out[-1]["class_display_name"] = "正常车辆"
                        tracks_out[-1]["label"] = "正常车辆"

                det["track_id"] = f"{det_track_prefix}-{track.track_id:03d}"
                det["track_num"] = int(track.track_id)
                det["frame_id"] = fid
                det["frame_ts"] = frame_ts
                det["detected_at"] = detect_time
                det["ocr_status"] = ocr_status
                det["plate_text"] = cached_text
                det["detector_model"] = detector_key
                det["model_display_label"] = str(model_info["display_label"])
                det["class_display_name"] = str(det.get("class_display_name", display_label) or display_label)
                det["display_label"] = display_label
                det["road_assignment"] = road_assignment
                det["road_id"] = road_assignment.get("road_id", "") if road_assignment else ""
                det["road_name"] = road_assignment.get("road_name", "") if road_assignment else ""
                det["road_segment"] = road_assignment.get("segment", "") if road_assignment else ""
                if detector_key == "normal" and det.get("normal_road_status") == "normal_vehicle":
                    det["class_display_name"] = "正常车辆"
                    det["display_label"] = "正常车辆"

            # 正常道路分析：只在用户选择的 normal ROI 内，把 car/vehicle 视为正常目标。
            normal_vehicle_boxes = [
                item for item in current_boxes_out
                if item.get("normal_road_status") == "normal_vehicle"
            ] if detector_key == "normal" else []
            normal_inside_other = [
                item for item in current_boxes_out
                if item.get("normal_road_status") == "inside_roi_unclassified"
            ] if detector_key == "normal" else []
            normal_road_analysis = {
                "enabled": detector_key == "normal",
                "roi_ready": bool(normal_roi_pixels),
                "normal_vehicle_count": len(normal_vehicle_boxes),
                "inside_unclassified_count": len(normal_inside_other),
                "stable_lane_count": int(normal_lane_info.get("stable_lane_count", 0) or 0),
                "message": (
                    f"ROI 内识别到 {len(normal_vehicle_boxes)} 辆正常车辆，已绘制 {int(normal_lane_info.get('stable_lane_count', 0) or 0)} 条稳定车道线。"
                    if detector_key == "normal"
                    else "当前模式未启用正常道路分析。"
                ),
            }

            # 禁停区域监测：stop.onnx 输出 car / carNumber / noParking 时，
            # 仅 car 与 noParking 参与禁停计时，连续 3 秒触发告警。
            parking_info = update_parking_monitor(current_boxes_out, w, h)
            annotate_parking_to_boxes(current_boxes_out, parking_info)
            annotate_parking_to_boxes(plates_out, parking_info)
            annotate_parking_to_boxes(tracks_out, parking_info)
            parking_by_track = {int(x.get("track_num", 0) or 0): x for x in (parking_info.get("active") or [])}
            for det in detections:
                try:
                    tn = int(det.get("track_num", 0) or 0)
                except Exception:
                    tn = 0
                p = parking_by_track.get(tn)
                if p and is_heatmap_vehicle_detection(det):
                    det["parking_in_zone"] = bool(p.get("in_zone"))
                    det["parking_alert"] = bool(p.get("alert"))
                    det["parking_dwell_s"] = float(p.get("dwell_s", 0.0) or 0.0)
                    det["parking_threshold_s"] = float(p.get("threshold_s", 3.0) or 3.0)
                    det["parking_zone_label"] = str(p.get("zone_label", "禁停区域"))
                    det["parking_event_id"] = str(p.get("event_id", ""))

            heatmap_boxes = [b for b in current_boxes_out if is_heatmap_vehicle_detection(b)]
            road_map = build_abstract_road_map(heatmap_boxes, w, h)
            track_ms = (time.perf_counter() - track_t0) * 1000.0
            perf_add("track_update_ms", track_ms)
            with STATE.lock:
                # 核心：latest_yolo_boxes 只保存本次 YOLO 的实时框。
                # 如果本帧没检测到，立即置空，避免旧框漂移。
                STATE.latest_yolo_boxes = current_boxes_out
                STATE.latest_detections = detections
                STATE.latest_result = {
                    "timestamp": time.time(),
                    "frame_id": fid,
                    "detections": detections,
                    "plates": plates_out,
                    "tracks": tracks_out,
                    "vehicle_count": int((road_map.get("summary") or {}).get("vehicle_count", len(heatmap_boxes))),
                    "frame_width": int(w),
                    "frame_height": int(h),
                    "overlay_mode": "frontend_canvas" if not STATE.config.server_overlay else "server_overlay",
                    "server_overlay": bool(STATE.config.server_overlay),
                    "detector_model": detector_key,
                    "model_display_label": str(model_info["display_label"]),
                    "normal_lane": normal_lane_info,
                    "normal_road_analysis": normal_road_analysis,
                    "road_map": road_map,
                    "road_assignments": road_map.get("assignments", []),
                    "road_heat": road_map.get("heat", []),
                    "parking_monitor": parking_info,
                    "parking_active": parking_info.get("active", []),
                    "parking_alerts": parking_info.get("alerts", []),
                    "parking_alert_count": len(parking_info.get("alerts", []) or []),
                    "message": (
                        normal_road_analysis["message"]
                        if detector_key == "normal"
                        else (
                            f"禁停告警：{len(parking_info.get('alerts', []) or [])} 辆车已停止在禁停区超过 {STATE.config.parking_violation_seconds:.0f} 秒"
                            if parking_info.get("alerts")
                            else (
                                f"禁停监测：{len(parking_info.get('active', []) or [])} 辆车在禁停区内，只有停车后才开始 {STATE.config.parking_violation_seconds:.0f} 秒计时"
                                if parking_info.get("active")
                                else (
                                    f"{model_info['display_label']} YOLO 实时框已更新；热力图只按当前 car/车辆目标实时刷新"
                                    if heatmap_boxes
                                    else (
                                        f"{model_info['display_label']} YOLO 实时框已更新；当前类别不进入车辆热力图"
                                        if current_boxes_out and detector_key == "stop"
                                        else (
                                            f"{model_info['display_label']} YOLO 实时框已更新"
                                            if current_boxes_out
                                            else f"当前帧未检测到{model_info['display_label']}"
                                        )
                                    )
                                )
                            )
                        )
                    ),
                }
                latency_ms = (time.time() - frame_ts) * 1000.0 if frame_ts else 0.0

            count += 1
            now3 = time.time()
            if now3 - fps_t >= 1.0:
                STATE.update_status(yolo_fps=round(count / max(1e-6, now3 - fps_t), 2))
                count = 0
                fps_t = now3
            with STATE.lock:
                pending_tracks = sum(1 for t in STATE.tracks if t.ocr_pending)
                qsize = STATE.ocr_queue.qsize()
            STATE.update_status(
                yolo_ms=round(yolo_ms, 2),
                track_ms=round(track_ms, 2),
                latency_ms=round(latency_ms, 2),
                ocr_queue_size=int(qsize),
                ocr_pending_tracks=int(pending_tracks),
                frame_age_ms=round((time.time() - frame_ts) * 1000.0, 2) if frame_ts else 0.0,
                detector_model=detector_key,
                active_model_name=str(model_info["name"]),
                model_display_label=str(model_info["display_label"]),
                parking_alerts=len(parking_info.get("alerts", []) or []),
                parking_active=len(parking_info.get("active", []) or []),
                normal_roi_ready=bool(normal_roi_pixels) if detector_key == "normal" else False,
                normal_lane_count=int(normal_lane_info.get("stable_lane_count", 0) or 0),
                normal_lane_ms=float(normal_lane_info.get("processing_ms", 0.0) or 0.0),
                normal_vehicle_count=len(normal_vehicle_boxes),
                parking_threshold_s=float(STATE.config.parking_violation_seconds),
            )
            maybe_log_perf()
        except Exception as exc:
            log(f"YOLO worker error: {exc}")
            log(traceback.format_exc())
            STATE.update_status(message=f"YOLO 推理错误：{exc}")
            time.sleep(0.1)

def overlay_frame(frame: np.ndarray) -> np.ndarray:
    """叠加显示层。

    正确逻辑：
    - 框：只画最新 YOLO 检测框 STATE.latest_yolo_boxes；
    - 字：根据 track_id 从对应 track 中读取 OCR 缓存，能复用就复用；
    - OCR 没回来也要画框，标签显示“识别中”。
    """
    display = frame.copy()
    now = time.time()
    with STATE.lock:
        latest_id = int(STATE.latest_frame_id)
        source_type = str(STATE.status.get("source_type", "idle"))
        max_lag = max_frame_lag_for_current_source()
        display_ttl = display_ttl_for_current_source()
        yolo_boxes = [dict(b) for b in STATE.latest_yolo_boxes]
        # live track cache：让 OCR 线程刚写入的文字能立刻复用到最新 YOLO 框上。
        track_cache: Dict[int, PlateTrack] = {int(t.track_id): t for t in STATE.tracks}
        status = dict(STATE.status)
        result_msg = STATE.latest_result.get("message", "")
        min_stable_votes = int(STATE.config.min_stable_votes)

    for box in yolo_boxes:
        box_frame_id = int(box.get("frame_id", 0) or 0)
        detected_at = float(box.get("detected_at", 0.0) or 0.0)
        if detected_at and now - detected_at > display_ttl:
            continue
        if box_frame_id and latest_id - box_frame_id > max_lag:
            continue

        bbox = box.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        track_num = int(box.get("track_num", 0) or 0)
        tr = track_cache.get(track_num)

        is_vehicle_box = (str(box.get("detector_model", "")) == "vehicle" or str(box.get("model_label", "")) == "vehicle")
        if is_vehicle_box:
            stable = False
            text = str(box.get("model_display_label", "车辆") or "车辆")
            score = float(box.get("det_confidence", 0.0) or 0.0)
            votes = 0
            pending = False
        elif tr is not None:
            stable = bool(tr.stable_text and len(tr.candidates) >= min_stable_votes)
            preview_text, preview_score = PlateVoting.preview(tr)
            text = tr.stable_text or preview_text
            score = tr.stable_score or preview_score or float(box.get("det_confidence", 0.0) or 0.0)
            votes = len(tr.candidates)
            pending = tr.ocr_pending
        else:
            stable = bool(box.get("stable", False))
            text = str(box.get("plate_text", "") or box.get("stable_text", "") or "")
            score = float(box.get("ocr_confidence", 0.0) or box.get("det_confidence", 0.0) or 0.0)
            votes = int(box.get("votes", 0) or 0)
            pending = bool(box.get("ocr_pending", False))

        # 没 OCR 文本也必须实时画框。车辆模型不走 OCR，只显示车辆框和置信度。
        color = (40, 230, 120) if stable else ((255, 180, 60) if is_vehicle_box else (0, 210, 255))
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        if is_vehicle_box:
            # 车辆/禁停等需要跟踪的模式仍可显示车辆编号。
            label = f"#{track_num} {text} | YOLO {score:.2f}"
            if pending:
                label += " OCR..."
        elif text:
            # 车牌识别模式只显示车牌号，不显示 track_id、编号、票数或 OCR 状态。
            label = str(text)
        else:
            label = "车牌识别中"
        display = draw_label(display, label, (x1, max(4, y1 - 30)), color)

    status_line = (
        f"{source_type.upper()} | 实时YOLO框+OCR缓存复用 | "
        f"CAP {status.get('capture_fps', 0):.1f} "
        f"YOLO {status.get('yolo_fps', 0):.1f}/{status.get('yolo_ms', 0):.0f}ms "
        f"OCR {status.get('ocr_fps', 0):.1f}/{status.get('ocr_ms', 0):.0f}ms "
        f"MOTION {status.get('motion_score', 0):.1f} "
        f"{status.get('yolo_provider', '')}"
    )
    display = draw_label(display, status_line, (12, 12), (72, 216, 255))
    if result_msg:
        display = draw_label(display, result_msg, (12, 42), (180, 230, 255))
    return display

def clear_runtime_state() -> None:
    with STATE.lock:
        STATE.latest_frame = None
        STATE.latest_frame_id = 0
        STATE.latest_frame_ts = 0.0
        STATE.source_finished = False
        STATE.motion_gray = None
        STATE.motion_score = 0.0
        STATE.tracks.clear()
        STATE.next_track_id = 1
        STATE.latest_yolo_boxes = []
        STATE.latest_detections = []
        STATE.latest_result = {"plates": [], "detections": [], "tracks": [], "parking_monitor": {"active": [], "alerts": [], "zones": []}, "parking_alert_count": 0, "message": "等待第一帧检测。"}
        STATE.normal_roi_normalized = []
        STATE.normal_lane_detector.reset()
        STATE.normal_lane_result = STATE.normal_lane_detector.empty_result(
            status="waiting_roi",
            message="正常道路模式需要先选择 ROI。",
        )
        STATE.parking_tracks.clear()
        STATE.parking_zones.clear()
        STATE.status["parking_alerts"] = 0
        STATE.status["parking_active"] = 0
        while True:
            try:
                STATE.ocr_queue.get_nowait()
            except Exception:
                break
        # 新工作流重新统计性能，避免上一段视频污染判断。
        STATE.perf_samples.clear()
        STATE.perf_counters.clear()
        STATE.perf_reset_at = time.time()

def stop_current_worker() -> None:
    STATE.stop_event.set()
    for th in [STATE.capture_thread, STATE.yolo_thread, STATE.ocr_thread]:
        if th and th.is_alive():
            th.join(timeout=2.0)
    STATE.capture_thread = None
    STATE.yolo_thread = None
    STATE.ocr_thread = None
    STATE.stop_event.clear()
    STATE.update_status(running=False)


def start_pipeline(
    source: str,
    source_type: str,
    normal_roi: Optional[List[List[float]]] = None,
) -> None:
    if STATE.ort_session is None or STATE.ocr_model is None:
        raise HTTPException(status_code=500, detail=f"模型未就绪：{STATE.status.get('message')}")

    detector_key = normalize_detector_model(getattr(STATE.config, "detector_model", "plate"))
    cleaned_normal_roi: List[List[float]] = []
    if detector_key == "normal":
        cleaned_normal_roi = sanitize_normalized_roi(normal_roi or [])
        if len(cleaned_normal_roi) < 3:
            raise HTTPException(
                status_code=400,
                detail="正常道路模式必须先在视频预览中选择至少 3 个 ROI 顶点，再点击开始检测。",
            )

    stop_current_worker()
    clear_runtime_state()

    if detector_key == "normal":
        STATE.normal_roi_normalized = cleaned_normal_roi
        STATE.normal_lane_detector.configure(cleaned_normal_roi)
        STATE.normal_lane_result = STATE.normal_lane_detector.empty_result(
            status="waiting",
            message="ROI 已确认，等待第一帧进行连续车道线检测。",
        )

    sync_mode = "手持防拖影" if source_type == "camera" else "固定/本地视频稳定显示"
    normal_note = "；正常道路 ROI 与连续车道线检测已启用" if detector_key == "normal" else ""
    STATE.update_status(
        running=True,
        source_type=source_type,
        source=source,
        sync_mode=sync_mode,
        normal_roi_ready=bool(cleaned_normal_roi),
        message=f"正在启动并行视频分析流水线（{sync_mode}）：当前模型自动适配输入尺寸，YOLO实时画框{normal_note}...",
    )
    STATE.capture_thread = threading.Thread(target=capture_loop, args=(source, source_type), daemon=True)
    STATE.yolo_thread = threading.Thread(target=yolo_loop, daemon=True)
    STATE.ocr_thread = threading.Thread(target=ocr_loop, daemon=True)
    STATE.capture_thread.start()
    STATE.yolo_thread.start()
    STATE.ocr_thread.start()


@app.on_event("startup")
def on_startup() -> None:
    init_system_database()
    threading.Thread(target=load_models, daemon=True).start()


@app.get("/api/status")
def api_status() -> Dict[str, Any]:
    return STATE.snapshot()


@app.get("/api/latest")
def api_latest() -> Dict[str, Any]:
    return STATE.snapshot()



@app.get("/api/perf")
def api_perf() -> Dict[str, Any]:
    return {"ok": True, "perf": get_perf_summary(), "status": dict(STATE.status)}


@app.post("/api/perf/reset")
def api_perf_reset() -> Dict[str, Any]:
    with STATE.lock:
        STATE.perf_samples.clear()
        STATE.perf_counters.clear()
        STATE.perf_reset_at = time.time()
    return {"ok": True, "message": "性能监测数据已清空"}


@app.post("/api/config")
def api_config(config: BackendConfig) -> Dict[str, Any]:
    """动态更新参数与检测模型。

    关键约束：切换 detector_model 时绝不调用 stop_current_worker，也不清空 latest_frame，
    因而手机 capture_loop、视频源和 MJPEG 浏览连接都保持不变。这里只短暂停止 YOLO 推理，
    原子替换 ONNX Session 后继续消费同一个视频读取线程产生的最新帧。
    """
    old_detector = normalize_detector_model(getattr(STATE.config, "detector_model", "plate"))
    new_detector = normalize_detector_model(config.detector_model)
    config.detector_model = new_detector
    detector_changed = (
        old_detector != new_detector
        or normalize_detector_model(STATE.status.get("detector_model")) != new_detector
    )

    # 先挂起 YOLO，再发布新配置，避免出现“新 detector key + 旧 Session”的瞬时组合。
    if detector_changed:
        STATE.model_switch_event.set()
        with STATE.lock:
            STATE.model_generation += 1
    STATE.config = config
    try:
        STATE.ocr_queue = queue.Queue(maxsize=config.max_ocr_queue)
    except Exception:
        pass

    if detector_changed:
        with STATE.model_switch_lock:
            STATE.update_status(
                model_switching=True,
                message=f"视频连接保持中，正在切换到 {active_model_info(new_detector)['display_label']} 模型...",
            )
            try:
                # 只清除旧模型的框、OCR 投票与禁停临时状态，不碰视频帧、source、capture_thread。
                reset_visual_state(f"正在切换检测模型：{active_model_info(new_detector)['display_label']}。")
                with STATE.inference_lock:
                    load_models()
            finally:
                STATE.model_switch_event.clear()
                STATE.update_status(model_switching=False)

    model_info = active_model_info(new_detector)
    STATE.update_status(
        server_overlay=bool(config.server_overlay),
        detector_model=new_detector,
        active_model_name=model_info["name"],
        model_label=model_info["label"],
        model_display_label=model_info["display_label"],
        model_uses_ocr=bool(model_info["uses_ocr"]),
        message=(
            f"已切换到 {model_info['display_label']}，继续使用当前视频连接。"
            if detector_changed and bool(STATE.status.get("running"))
            else f"参数已更新，当前检测模型：{model_info['display_label']}。"
        ),
    )
    return {
        "ok": True,
        "config": safe_model_dump(config),
        "detector_model": new_detector,
        "model_path": STATE.status.get("model_path"),
        "stream_preserved": True,
        "running": bool(STATE.status.get("running")),
        "source": str(STATE.status.get("source", "")),
    }


def _read_normal_roi_preview(source: str, source_type: str) -> np.ndarray:
    source_type = str(source_type or "file").lower()
    if source_type == "file":
        path = Path(source)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"视频文件不存在：{source}")

    cap = cv2.VideoCapture(source)
    if source_type == "camera":
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"无法打开 ROI 预览来源：{source}")

    frame = None
    # 网络流有时第一帧为空，轻量重试；本地视频通常第一次即可。
    for _ in range(20 if source_type == "camera" else 3):
        ok, candidate = cap.read()
        if ok and candidate is not None and candidate.size:
            frame = candidate
            break
        time.sleep(0.05)
    cap.release()

    if frame is None:
        raise HTTPException(status_code=400, detail="无法读取 ROI 预览帧。")

    max_width = 1280
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / float(width)
        frame = cv2.resize(
            frame,
            (max_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return frame


@app.post("/api/normal/roi/preview")
def api_normal_roi_preview(req: NormalRoiPreviewRequest) -> Dict[str, Any]:
    # 优先复用正在运行的 capture_loop 最新帧，避免为了圈 ROI 再次连接手机 IP Webcam。
    source = str(req.source or "").strip()
    source_type = str(req.source_type or "file").lower()
    if not source:
        raise HTTPException(status_code=400, detail="ROI 预览来源不能为空。")

    frame = None
    reused_live_frame = False
    with STATE.lock:
        same_running_source = (
            bool(STATE.status.get("running"))
            and str(STATE.status.get("source", "")) == source
            and str(STATE.status.get("source_type", "idle")) == source_type
            and STATE.latest_frame is not None
        )
        if same_running_source:
            frame = STATE.latest_frame.copy()
            reused_live_frame = True

    if frame is None:
        frame = _read_normal_roi_preview(source, source_type)

    max_width = 1280
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / float(width)
        frame = cv2.resize(
            frame,
            (max_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 86],
    )
    if not ok:
        raise HTTPException(status_code=500, detail="ROI 预览帧编码失败。")
    data_url = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    return {
        "ok": True,
        "source_type": source_type,
        "frame_width": int(frame.shape[1]),
        "frame_height": int(frame.shape[0]),
        "image_data_url": data_url,
        "reused_live_frame": reused_live_frame,
        "message": "已复用当前视频帧，请沿道路边界依次点击至少 3 个点。" if reused_live_frame else "请沿正常道路边界依次点击至少 3 个点。",
    }


@app.post("/api/normal/roi/configure")
def api_normal_roi_configure(req: NormalRoiConfigureRequest) -> Dict[str, Any]:
    """运行中更新道路 ROI，不重启或重连视频源。"""
    cleaned = sanitize_normalized_roi(req.points or [])
    if len(cleaned) < 3:
        raise HTTPException(status_code=400, detail="道路 ROI 至少需要 3 个有效顶点。")

    with STATE.lock:
        STATE.normal_roi_normalized = cleaned
        STATE.normal_lane_detector.configure(cleaned)
        STATE.normal_lane_result = STATE.normal_lane_detector.empty_result(
            status="waiting",
            message="道路 ROI 已更新，等待下一帧分析。",
        )
        STATE.status["normal_roi_ready"] = True

    reset_visual_state("道路 ROI 已更新，继续使用当前视频连接。")
    return {
        "ok": True,
        "points": cleaned,
        "running": bool(STATE.status.get("running")),
        "source": str(STATE.status.get("source", "")),
        "stream_preserved": True,
    }


@app.post("/api/start/video")
def api_start_video(req: StartVideoRequest) -> Dict[str, Any]:
    path = Path(req.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"视频文件不存在：{req.path}")
    start_pipeline(str(path), "file", req.normal_roi)
    return {"ok": True, "message": "已启动本地视频 ONNX 并行检测工作流", "source": str(path)}


@app.post("/api/start/camera")
def api_start_camera(req: StartCameraRequest) -> Dict[str, Any]:
    if not req.url:
        raise HTTPException(status_code=400, detail="请填写 IP Webcam 视频流地址")
    start_pipeline(req.url, "camera", req.normal_roi)
    return {"ok": True, "message": "已启动手机视频流 ONNX 并行检测工作流", "source": req.url}


@app.post("/api/stop")
def api_stop() -> Dict[str, Any]:
    stop_current_worker()
    clear_runtime_state()
    STATE.update_status(source_type="idle", source="", message="已停止当前工作流。")
    return {"ok": True, "message": "已停止"}


@app.get("/api/stream.mjpg")
def api_stream() -> StreamingResponse:
    def gen():
        display_count = 0
        fps_t = time.time()
        while True:
            with STATE.lock:
                frame = None if STATE.latest_frame is None else STATE.latest_frame.copy()
                msg = STATE.status.get("message", "等待视频输入...")
                quality = int(STATE.config.jpeg_quality)
                sleep_s = 1.0 / max(1.0, float(STATE.config.display_fps))
            if frame is None:
                frame = np.zeros((480, 800, 3), dtype=np.uint8)
                frame[:] = (7, 17, 31)
                frame = draw_label(frame, msg, (24, 40), (72, 216, 255))
                overlay_ms = 0.0
                perf_add("overlay_ms", overlay_ms)
                STATE.update_status(overlay_ms=round(overlay_ms, 2))
            else:
                # V4 速度优化：默认不在 Python 后端画框/中文，避免 PIL/OpenCV overlay 成为瓶颈。
                # 检测框与 OCR 文本通过 /api/latest 给前端 Canvas 叠加。
                overlay_t0 = time.perf_counter()
                if bool(STATE.config.server_overlay):
                    frame = overlay_frame(frame)
                overlay_ms = (time.perf_counter() - overlay_t0) * 1000.0
                perf_add("overlay_ms", overlay_ms)
                STATE.update_status(overlay_ms=round(overlay_ms, 2), server_overlay=bool(STATE.config.server_overlay))

            enc_t0 = time.perf_counter()
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            encode_ms = (time.perf_counter() - enc_t0) * 1000.0
            perf_add("jpeg_encode_ms", encode_ms)
            STATE.update_status(encode_ms=round(encode_ms, 2))
            data = buffer.tobytes() if ok else b""
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
            display_count += 1
            now = time.time()
            if now - fps_t >= 1.0:
                STATE.update_status(display_fps=round(display_count / max(1e-6, now - fps_t), 2))
                display_count = 0
                fps_t = now
            time.sleep(sleep_s)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")



@app.post("/api/auth/register")
def api_auth_register(req: RegisterRequest) -> Dict[str, Any]:
    username = req.username.strip()
    if len(username) < 2 or len(req.password) < 4:
        raise HTTPException(status_code=400, detail="用户名至少 2 位，密码至少 4 位")
    salt, digest = hash_password(req.password)
    try:
        db_exec("INSERT INTO users(username,password_hash,salt,role,display_name,created_at) VALUES(?,?,?,?,?,?)",
                (username, digest, salt, req.role, req.display_name or username, time.strftime('%Y-%m-%d %H:%M:%S')))
        log_operation('register_user', f'新增用户 {username} / {req.role}', username)
        return {"ok": True, "message": "用户注册成功", "username": username, "role": req.role}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="用户名已存在")


@app.post("/api/auth/login")
def api_auth_login(req: LoginRequest) -> Dict[str, Any]:
    rows = db_query("SELECT * FROM users WHERE username=?", (req.username.strip(),))
    if not rows or not verify_password(req.password, rows[0]['salt'], rows[0]['password_hash']):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    db_exec("UPDATE users SET last_login=? WHERE username=?", (now, req.username.strip()))
    log_operation('login', '用户登录系统', req.username.strip())
    return {"ok": True, "message": "登录成功", "user": {"username": rows[0]['username'], "role": rows[0]['role'], "display_name": rows[0]['display_name']}}


@app.get("/api/users")
def api_users() -> Dict[str, Any]:
    rows = db_query("SELECT user_id,username,role,display_name,created_at,last_login FROM users ORDER BY user_id DESC LIMIT 50")
    return {"ok": True, "users": rows}


@app.get("/api/devices")
def api_devices() -> Dict[str, Any]:
    return {"ok": True, "devices": db_query("SELECT * FROM video_sources ORDER BY source_id DESC")}


@app.post("/api/devices")
def api_device_add(req: DeviceRequest) -> Dict[str, Any]:
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    db_exec("INSERT INTO video_sources(name,type,stream_url,position,status,note,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
            (req.name, req.type, req.stream_url, req.position, req.status, req.note, now, now))
    log_operation('add_device', f'{req.name} / {req.type} / {req.position}')
    return {"ok": True, "message": "设备/视频源已登记"}


@app.get("/api/model-catalog")
def api_model_catalog() -> Dict[str, Any]:
    rows = db_query("SELECT * FROM model_configs ORDER BY model_id")
    status = dict(STATE.status)
    return {"ok": True, "models": rows, "runtime": {"active_model": status.get('detector_model'), "provider": status.get('yolo_provider'), "model_path": status.get('model_path')}}


@app.get("/api/whitelist")
def api_whitelist() -> Dict[str, Any]:
    return {"ok": True, "items": db_query("SELECT * FROM plate_whitelist ORDER BY id DESC")}


@app.post("/api/whitelist")
def api_whitelist_add(req: WhitelistRequest) -> Dict[str, Any]:
    plate_no = clean_plate_text(req.plate_no)
    if not plate_no:
        raise HTTPException(status_code=400, detail="车牌号不能为空")
    db_exec("INSERT OR REPLACE INTO plate_whitelist(plate_no,owner,allow,note,created_at) VALUES(?,?,?,?,?)",
            (plate_no, req.owner, 1 if req.allow else 0, req.note, time.strftime('%Y-%m-%d %H:%M:%S')))
    log_operation('upsert_whitelist', f'{plate_no} / {req.owner}')
    return {"ok": True, "message": "白名单已保存", "plate_no": plate_no}


@app.get("/api/system/configs")
def api_system_configs() -> Dict[str, Any]:
    return {"ok": True, "configs": db_query("SELECT * FROM system_configs ORDER BY config_key")}


@app.post("/api/system/configs")
def api_system_config_set(req: ConfigItemRequest) -> Dict[str, Any]:
    db_exec("INSERT OR REPLACE INTO system_configs(config_key,config_value,description,updated_at) VALUES(?,?,?,?)",
            (req.key, req.value, req.description, time.strftime('%Y-%m-%d %H:%M:%S')))
    log_operation('set_config', f'{req.key}={req.value}')
    return {"ok": True, "message": "配置已保存"}


@app.get("/api/history")
def api_history(type: str = "all") -> Dict[str, Any]:
    parking_history: List[Dict[str, Any]] = []
    try:
        if PARKING_EVENT_LOG_PATH.exists():
            with PARKING_EVENT_LOG_PATH.open('r', encoding='utf-8') as f:
                for line in f:
                    line=line.strip()
                    if line:
                        parking_history.append(json.loads(line))
    except Exception:
        parking_history = []
    plate_rows = db_query("SELECT * FROM plate_records ORDER BY record_id DESC LIMIT 30")
    traffic_rows = db_query("SELECT * FROM traffic_stats ORDER BY stat_id DESC LIMIT 30")
    ops = db_query("SELECT * FROM operation_logs ORDER BY log_id DESC LIMIT 50")
    return {"ok": True, "plate_records": plate_rows, "traffic_stats": traffic_rows, "parking_events": parking_history[-50:][::-1], "operation_logs": ops}


@app.get("/api/admin/summary")
def api_admin_summary() -> Dict[str, Any]:
    users = db_query("SELECT user_id,username,role,display_name,last_login FROM users ORDER BY user_id")
    devices = db_query("SELECT * FROM video_sources ORDER BY source_id")
    models = db_query("SELECT * FROM model_configs ORDER BY model_id")
    whitelist = db_query("SELECT * FROM plate_whitelist ORDER BY id DESC LIMIT 20")
    configs = db_query("SELECT * FROM system_configs ORDER BY config_key")
    logs = db_query("SELECT * FROM operation_logs ORDER BY log_id DESC LIMIT 10")
    history = api_history()
    return {"ok": True, "users": users, "devices": devices, "models": models, "whitelist": whitelist, "configs": configs, "logs": logs, "history": history}

@app.get("/")
def root() -> Dict[str, str]:
    return {"service": "TrafficVision Plate Backend ONNX Parallel", "status": STATE.status.get("backend", "unknown")}


if __name__ == "__main__":
    import uvicorn

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    log(f"Starting backend on 127.0.0.1:{BACKEND_PORT}")
    uvicorn.run(app, host="127.0.0.1", port=BACKEND_PORT, log_level="warning", access_log=False)




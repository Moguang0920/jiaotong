# -*- coding: utf-8 -*-
"""
智慧交通视觉感知系统 - ONNX 并行版车牌检测后端

核心改进：
1. 车牌检测优先加载 best(1).onnx，不再使用 best(1).pt 串行推理。
2. 三线程解耦：视频读取线程、YOLO ONNX 推理线程、PaddleOCR 识别线程。
3. 前端 MJPEG 显示不等待 AI 推理，始终展示最新视频帧 + 最近一次 AI 结果。
4. OCR 不再每帧执行；每个 track_id 每秒最多补充 1 次 OCR，但会连续多次投票，直到稳定。
5. OCR 默认使用 PP-OCRv6_tiny_rec，失败时自动回退默认 TextRecognition。

模型放置：
- 推荐：D:/pycharm/Jiaotong-gpt/best(1).onnx
- 备用：环境变量 PLATE_ONNX_MODEL 指定 ONNX 路径

启动：
- 由 Electron main.js 自动启动，或手动执行：
  python backend/plate_runtime_backend.py
"""

from __future__ import annotations

import json
import os
import queue
import re
import sys
import time
import threading
import traceback
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
DEFAULT_CAMERA_URL = os.environ.get("IP_WEBCAM_URL", "http://100.70.11.30:8080/video")
BACKEND_PORT = int(os.environ.get("TRAFFIC_BACKEND_PORT", "8765"))

PROVINCES = set("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼")
SPECIAL_PLATE_PREFIXES = {"使", "领", "警", "学", "港", "澳"}
VALID_PREFIXES = PROVINCES | SPECIAL_PLATE_PREFIXES


class StartVideoRequest(BaseModel):
    path: str


class StartCameraRequest(BaseModel):
    url: str = DEFAULT_CAMERA_URL


class BackendConfig(BaseModel):
    # ONNX 推理参数
    conf: float = 0.35
    imgsz: int = 416
    nms_iou: float = 0.45
    yolo_interval: float = 0.12  # 每 0.12 秒最多跑一次 YOLO，约 8 FPS

    # OCR 参数：不是每帧 OCR，而是同一个 track 每秒最多补充一次，连续多次投票
    ocr_min_interval: float = 1.0
    stable_recheck_interval: float = 2.0
    min_stable_votes: int = 3
    max_vote_history: int = 12
    min_ocr_conf: float = 0.45
    ocr_model_name: str = "PP-OCRv6_tiny_rec"

    # 显示参数
    display_fps: float = 20.0
    jpeg_quality: int = 78
    max_ocr_queue: int = 8


@dataclass
class PlateCandidate:
    raw: str
    cleaned: str
    confidence: float
    timestamp: float
    quality: float
    province: str = ""
    body: str = ""


@dataclass
class PlateTrack:
    track_id: int
    bbox: Tuple[int, int, int, int]
    det_conf: float
    last_seen: float
    candidates: List[PlateCandidate] = field(default_factory=list)
    stable_text: str = ""
    stable_score: float = 0.0
    stable_at: float = 0.0
    last_ocr_time: float = 0.0
    ocr_pending: bool = False
    ocr_attempts: int = 0

    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def is_recent(self, now: float, ttl: float = 2.0) -> bool:
        return now - self.last_seen <= ttl


class PlateVoting:
    """多帧投票与车牌纠错。

    目标：把京E4682Y / E4682Y / RE4682Y / 康E4682Y 这类波动，稳定到最合理结果。
    思路：完整车牌票 + 省份票 + 主体票 + OCR 置信度权重。
    """

    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""
        text = text.upper().replace("·", "").replace(".", "").replace(" ", "")
        return re.sub(r"[^0-9A-Z\u4e00-\u9fa5]", "", text)

    @staticmethod
    def split_plate(text: str) -> Tuple[str, str, float]:
        cleaned = PlateVoting.clean(text)
        if not cleaned:
            return "", "", 0.0

        province = ""
        body = ""
        quality = 0.35

        first = cleaned[0]
        if first in VALID_PREFIXES:
            province = first
            body = re.sub(r"[^0-9A-Z]", "", cleaned[1:])
            quality = 1.0 if 5 <= len(body) <= 7 else 0.75
        else:
            alnum = re.sub(r"[^0-9A-Z]", "", cleaned)
            if len(alnum) >= 6:
                body = alnum[-6:]
                quality = 0.70
            elif alnum:
                body = alnum
                quality = 0.45

        if body:
            has_digit = any(ch.isdigit() for ch in body)
            has_alpha = any("A" <= ch <= "Z" for ch in body)
            if not (has_digit and has_alpha):
                quality *= 0.72
        return province, body, quality

    @staticmethod
    def make_candidate(raw_text: str, confidence: float) -> Optional[PlateCandidate]:
        cleaned = PlateVoting.clean(raw_text)
        if not cleaned:
            return None
        province, body, quality = PlateVoting.split_plate(cleaned)
        if not body:
            return None
        return PlateCandidate(
            raw=raw_text,
            cleaned=cleaned,
            confidence=float(confidence or 0.0),
            timestamp=time.time(),
            quality=quality,
            province=province,
            body=body,
        )

    @staticmethod
    def vote(track: PlateTrack, min_votes: int = 3) -> Tuple[str, float, Dict[str, Any]]:
        now = time.time()
        valid = [c for c in track.candidates if now - c.timestamp <= 20.0]
        if not valid:
            return "", 0.0, {"votes": 0}

        body_scores: Dict[str, float] = {}
        province_scores: Dict[str, float] = {}
        complete_scores: Dict[str, float] = {}
        body_counts: Dict[str, int] = {}
        province_counts: Dict[str, int] = {}
        complete_counts: Dict[str, int] = {}

        for c in valid:
            weight = max(0.01, c.confidence) * max(0.1, c.quality)
            if c.body:
                body_scores[c.body] = body_scores.get(c.body, 0.0) + weight
                body_counts[c.body] = body_counts.get(c.body, 0) + 1
            if c.province:
                province_scores[c.province] = province_scores.get(c.province, 0.0) + weight
                province_counts[c.province] = province_counts.get(c.province, 0) + 1
            if c.province and c.body:
                full = f"{c.province}{c.body}"
                complete_scores[full] = complete_scores.get(full, 0.0) + weight * 1.18
                complete_counts[full] = complete_counts.get(full, 0) + 1

        best_body = max(body_scores, key=body_scores.get) if body_scores else ""
        best_prov = max(province_scores, key=province_scores.get) if province_scores else ""
        best_complete = max(complete_scores, key=complete_scores.get) if complete_scores else ""

        debug = {
            "votes": len(valid),
            "best_body": best_body,
            "best_province": best_prov,
            "body_counts": body_counts,
            "province_counts": province_counts,
            "complete_counts": complete_counts,
        }

        if best_complete:
            score = complete_scores[best_complete] / max(1, len(valid))
            if complete_counts.get(best_complete, 0) >= min_votes or score >= 0.88:
                return best_complete, min(0.999, score), debug

        if best_body and best_prov:
            combined = f"{best_prov}{best_body}"
            score = (body_scores[best_body] + province_scores[best_prov]) / (2.0 * max(1, len(valid)))
            if body_counts.get(best_body, 0) >= min_votes or score >= 0.84:
                return combined, min(0.999, score), debug

        if best_complete:
            return best_complete, min(0.86, complete_scores[best_complete] / max(1, len(valid))), debug
        if best_body:
            return best_body, min(0.75, body_scores[best_body] / max(1, len(valid))), debug
        return "", 0.0, debug


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
            "display_fps": 0.0,
            "capture_fps": 0.0,
            "yolo_fps": 0.0,
            "ocr_fps": 0.0,
            "yolo_ms": 0.0,
            "ocr_ms": 0.0,
            "latency_ms": 0.0,
            "frame_id": 0,
            "model_path": str(DEFAULT_ONNX_PATH),
            "yolo_provider": "unknown",
            "ocr_model": self.config.ocr_model_name,
        }

        self.ort_session: Any = None
        self.ort_input_name: str = ""
        self.ort_output_names: List[str] = []
        self.ocr_model: Any = None

        self.stop_event = threading.Event()
        self.capture_thread: Optional[threading.Thread] = None
        self.yolo_thread: Optional[threading.Thread] = None
        self.ocr_thread: Optional[threading.Thread] = None
        self.ocr_queue: queue.Queue = queue.Queue(maxsize=self.config.max_ocr_queue)

        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_id: int = 0
        self.latest_frame_ts: float = 0.0
        self.source_finished: bool = False

        self.tracks: List[PlateTrack] = []
        self.next_track_id = 1
        self.latest_detections: List[Dict[str, Any]] = []
        self.latest_result: Dict[str, Any] = {"plates": [], "detections": [], "tracks": [], "message": "等待启动工作流。"}

    def update_status(self, **kwargs: Any) -> None:
        with self.lock:
            self.status.update(kwargs)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "status": dict(self.status),
                "config": self.config.dict(),
                "result": json.loads(json.dumps(self.latest_result, ensure_ascii=False)),
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


def log(msg: str) -> None:
    print(f"[PlateBackend] {msg}", flush=True)


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


def load_models() -> None:
    STATE.update_status(backend="loading", message="正在预加载 ONNX 车牌检测模型与轻量 OCR 识别模型。")
    errors: List[str] = []

    try:
        if ort is None:
            raise RuntimeError("未安装 onnxruntime / onnxruntime-gpu")
        if not DEFAULT_ONNX_PATH.exists():
            raise FileNotFoundError(f"未找到 ONNX 模型：{DEFAULT_ONNX_PATH}")

        available = ort.get_available_providers()
        providers: List[Any] = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        log(f"Loading ONNX model: {DEFAULT_ONNX_PATH}")
        log(f"ONNX available providers: {available}")
        log(f"ONNX using providers: {providers}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        STATE.ort_session = ort.InferenceSession(str(DEFAULT_ONNX_PATH), sess_options=sess_options, providers=providers)
        STATE.ort_input_name = STATE.ort_session.get_inputs()[0].name
        STATE.ort_output_names = [o.name for o in STATE.ort_session.get_outputs()]
        used_provider = STATE.ort_session.get_providers()[0] if STATE.ort_session.get_providers() else "unknown"
        STATE.update_status(yolo_ready=True, yolo_provider=used_provider, model_path=str(DEFAULT_ONNX_PATH))
    except Exception as exc:
        errors.append(f"ONNX加载失败: {exc}")
        log(traceback.format_exc())
        STATE.update_status(yolo_ready=False)

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

    ready = bool(STATE.ort_session is not None and STATE.ocr_model is not None)
    STATE.update_status(
        backend="ready" if ready else "degraded",
        models_ready=ready,
        message="ONNX 与 OCR 模型预加载完成。" if ready else "；".join(errors),
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


def parse_yolo_output(output: np.ndarray, frame_shape: Tuple[int, int], ratio: float, dwdh: Tuple[float, float], imgsz: int, conf_thres: float, iou_thres: float) -> List[Dict[str, Any]]:
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
    for idx in keep:
        x1, y1, x2, y2 = boxes[idx].tolist()
        detections.append({
            "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
            "confidence": round(float(scores[idx]), 4),
            "class_id": int(cls_list[idx]),
            "label": "plate",
        })
    return detections


def detect_plates_onnx(frame: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
    if STATE.ort_session is None:
        return [], 0.0
    cfg = STATE.config
    t0 = time.perf_counter()
    inp, ratio, dwdh = preprocess_for_onnx(frame, cfg.imgsz)
    outputs = STATE.ort_session.run(STATE.ort_output_names or None, {STATE.ort_input_name: inp})
    infer_ms = (time.perf_counter() - t0) * 1000.0
    detections = parse_yolo_output(outputs[0], frame.shape[:2], ratio, dwdh, cfg.imgsz, cfg.conf, cfg.nms_iou)
    return detections, infer_ms


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


def match_track(bbox: Tuple[int, int, int, int], det_conf: float) -> PlateTrack:
    now = time.time()
    with STATE.lock:
        STATE.tracks = [t for t in STATE.tracks if t.is_recent(now, ttl=2.2)]
        best: Optional[PlateTrack] = None
        best_score = 0.0
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        for tr in STATE.tracks:
            tcx, tcy = tr.center()
            dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
            box_iou = iou(tr.bbox, bbox)
            score = box_iou + max(0.0, 1.0 - dist / 160.0) * 0.35
            if score > best_score:
                best_score = score
                best = tr
        if best is not None and best_score >= 0.25:
            best.bbox = bbox
            best.det_conf = det_conf
            best.last_seen = now
            return best
        new_track = PlateTrack(track_id=STATE.next_track_id, bbox=bbox, det_conf=det_conf, last_seen=now)
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
    raw, conf = extract_text_recognition_result(output)
    return PlateVoting.clean(raw), float(conf or 0.0), ocr_ms


def should_schedule_ocr(track: PlateTrack, now: float) -> bool:
    cfg = STATE.config
    if track.ocr_pending:
        return False
    if now - track.last_ocr_time < cfg.ocr_min_interval:
        return False
    # 没稳定：每秒补一次，累计多次投票。
    if not track.stable_text:
        return True
    # 已稳定但票数还少：继续补充几次，增强稳定性。
    if len(track.candidates) < cfg.max_vote_history:
        return True
    # 已稳定后每隔一段时间重新校验一次。
    if now - track.last_ocr_time >= cfg.stable_recheck_interval:
        return True
    return False


def schedule_ocr(track: PlateTrack, frame: np.ndarray) -> None:
    crop = safe_crop(frame, track.bbox)
    if crop is None:
        return
    with STATE.lock:
        track.ocr_pending = True
        track.last_ocr_time = time.time()
        track.ocr_attempts += 1
    item = {"track_id": track.track_id, "crop": crop, "timestamp": time.time()}
    try:
        STATE.ocr_queue.put_nowait(item)
    except queue.Full:
        try:
            STATE.ocr_queue.get_nowait()
        except Exception:
            pass
        try:
            STATE.ocr_queue.put_nowait(item)
        except Exception:
            with STATE.lock:
                track.ocr_pending = False


def add_ocr_candidate(track_id: int, text: str, conf: float) -> None:
    with STATE.lock:
        track = next((t for t in STATE.tracks if t.track_id == track_id), None)
        if not track:
            return
        track.ocr_pending = False
        if conf < STATE.config.min_ocr_conf or not text:
            return
        candidate = PlateVoting.make_candidate(text, conf)
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
        try:
            text, conf, ocr_ms = recognize_plate(crop)
            add_ocr_candidate(track_id, text, conf)
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
        ret, frame = cap.read()
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
        with STATE.lock:
            STATE.latest_frame = frame
            STATE.latest_frame_id = fid
            STATE.latest_frame_ts = time.time()
            STATE.status["frame_id"] = fid
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
    log("YOLO ONNX worker started.")
    last_processed_id = -1
    last_infer_t = 0.0
    count = 0
    fps_t = time.time()

    while not STATE.stop_event.is_set():
        cfg = STATE.config
        now = time.time()
        if now - last_infer_t < cfg.yolo_interval:
            time.sleep(0.01)
            continue

        with STATE.lock:
            frame = None if STATE.latest_frame is None else STATE.latest_frame.copy()
            fid = STATE.latest_frame_id
            frame_ts = STATE.latest_frame_ts
        if frame is None or fid == last_processed_id:
            time.sleep(0.02)
            continue

        last_processed_id = fid
        last_infer_t = now
        try:
            detections, yolo_ms = detect_plates_onnx(frame)
            now2 = time.time()
            tracks_out: List[Dict[str, Any]] = []
            plates_out: List[Dict[str, Any]] = []
            for det in detections:
                bbox = tuple(int(v) for v in det["bbox"])
                track = match_track(bbox, float(det.get("confidence", 0.0)))
                if should_schedule_ocr(track, now2):
                    schedule_ocr(track, frame)

            with STATE.lock:
                # 输出当前活跃 track，不只输出本帧 detection，保证 OCR 投票结果能持续显示。
                active_tracks = [t for t in STATE.tracks if t.is_recent(time.time(), ttl=2.0)]
                for tr in active_tracks:
                    stable_text, stable_score, debug = PlateVoting.vote(tr, STATE.config.min_stable_votes)
                    if stable_text and (not tr.stable_text or stable_score >= tr.stable_score * 0.92):
                        tr.stable_text = stable_text
                        tr.stable_score = stable_score
                    label = tr.stable_text or (tr.candidates[-1].cleaned if tr.candidates else "检测到车牌")
                    stable = bool(tr.stable_text and len(tr.candidates) >= STATE.config.min_stable_votes)
                    plates_out.append({
                        "track_id": f"PLATE-{tr.track_id:03d}",
                        "plate_text": label,
                        "stable_text": tr.stable_text,
                        "raw_text": tr.candidates[-1].raw if tr.candidates else "",
                        "ocr_confidence": round(float(tr.candidates[-1].confidence), 4) if tr.candidates else 0.0,
                        "stable_score": round(float(tr.stable_score), 4),
                        "det_confidence": round(float(tr.det_conf), 4),
                        "bbox": list(tr.bbox),
                        "stable": stable,
                        "votes": len(tr.candidates),
                        "ocr_pending": tr.ocr_pending,
                        "vote_debug": debug,
                    })
                    tracks_out.append({
                        "track_id": f"PLATE-{tr.track_id:03d}",
                        "bbox": list(tr.bbox),
                        "label": label,
                        "stable": stable,
                        "votes": len(tr.candidates),
                    })

                STATE.latest_detections = detections
                STATE.latest_result = {
                    "timestamp": time.time(),
                    "frame_id": fid,
                    "detections": detections,
                    "plates": plates_out,
                    "tracks": tracks_out,
                    "vehicle_count": len(detections),
                    "message": "已检测到车牌" if plates_out else "未检测到车牌",
                }
                latency_ms = (time.time() - frame_ts) * 1000.0 if frame_ts else 0.0
            count += 1
            now3 = time.time()
            if now3 - fps_t >= 1.0:
                STATE.update_status(yolo_fps=round(count / max(1e-6, now3 - fps_t), 2))
                count = 0
                fps_t = now3
            STATE.update_status(yolo_ms=round(yolo_ms, 2), latency_ms=round(latency_ms, 2))
        except Exception as exc:
            log(f"YOLO worker error: {exc}")
            log(traceback.format_exc())
            STATE.update_status(message=f"YOLO 推理错误：{exc}")
            time.sleep(0.1)


def overlay_frame(frame: np.ndarray) -> np.ndarray:
    display = frame.copy()
    with STATE.lock:
        tracks = [t for t in STATE.tracks if t.is_recent(time.time(), ttl=2.0)]
        status = dict(STATE.status)
        result_msg = STATE.latest_result.get("message", "")
    for tr in tracks:
        x1, y1, x2, y2 = tr.bbox
        stable = bool(tr.stable_text and len(tr.candidates) >= STATE.config.min_stable_votes)
        text = tr.stable_text or (tr.candidates[-1].cleaned if tr.candidates else "检测到车牌")
        score = tr.stable_score or (tr.candidates[-1].confidence if tr.candidates else tr.det_conf)
        color = (40, 230, 120) if stable else (0, 210, 255)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        label = f"#{tr.track_id} {text} {score:.2f} | {len(tr.candidates)}票"
        if tr.ocr_pending:
            label += " OCR..."
        display = draw_label(display, label, (x1, max(4, y1 - 30)), color)

    status_line = (
        f"{status.get('source_type', 'IDLE').upper()} | CAP {status.get('capture_fps', 0):.1f} "
        f"YOLO {status.get('yolo_fps', 0):.1f}/{status.get('yolo_ms', 0):.0f}ms "
        f"OCR {status.get('ocr_fps', 0):.1f}/{status.get('ocr_ms', 0):.0f}ms "
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
        STATE.tracks.clear()
        STATE.next_track_id = 1
        STATE.latest_detections = []
        STATE.latest_result = {"plates": [], "detections": [], "tracks": [], "message": "等待第一帧检测。"}
        while True:
            try:
                STATE.ocr_queue.get_nowait()
            except Exception:
                break


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


def start_pipeline(source: str, source_type: str) -> None:
    if STATE.ort_session is None or STATE.ocr_model is None:
        raise HTTPException(status_code=500, detail=f"模型未就绪：{STATE.status.get('message')}")
    stop_current_worker()
    clear_runtime_state()
    STATE.update_status(running=True, source_type=source_type, source=source, message="正在启动并行视频分析流水线...")
    STATE.capture_thread = threading.Thread(target=capture_loop, args=(source, source_type), daemon=True)
    STATE.yolo_thread = threading.Thread(target=yolo_loop, daemon=True)
    STATE.ocr_thread = threading.Thread(target=ocr_loop, daemon=True)
    STATE.capture_thread.start()
    STATE.yolo_thread.start()
    STATE.ocr_thread.start()


@app.on_event("startup")
def on_startup() -> None:
    threading.Thread(target=load_models, daemon=True).start()


@app.get("/api/status")
def api_status() -> Dict[str, Any]:
    return STATE.snapshot()


@app.get("/api/latest")
def api_latest() -> Dict[str, Any]:
    return STATE.snapshot()


@app.post("/api/config")
def api_config(config: BackendConfig) -> Dict[str, Any]:
    # 运行中也可以动态调整检测频率、尺寸、阈值；OCR 模型名仅在重启后生效。
    STATE.config = config
    try:
        STATE.ocr_queue = queue.Queue(maxsize=config.max_ocr_queue)
    except Exception:
        pass
    return {"ok": True, "config": config.dict()}


@app.post("/api/start/video")
def api_start_video(req: StartVideoRequest) -> Dict[str, Any]:
    path = Path(req.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"视频文件不存在：{req.path}")
    start_pipeline(str(path), "file")
    return {"ok": True, "message": "已启动本地视频 ONNX 并行检测工作流", "source": str(path)}


@app.post("/api/start/camera")
def api_start_camera(req: StartCameraRequest) -> Dict[str, Any]:
    if not req.url:
        raise HTTPException(status_code=400, detail="请填写 IP Webcam 视频流地址")
    start_pipeline(req.url, "camera")
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
            else:
                frame = overlay_frame(frame)

            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
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
    uvicorn.run(app, host="127.0.0.1", port=BACKEND_PORT, log_level="info")

"""
Vehicle tracker module: IoU-based tracker with dwell-time alerts.
"""
from dataclasses import dataclass, field
import time
from typing import List, Tuple, Dict, Any, Optional

@dataclass
class VehicleTrack:
    track_id: int
    bbox: Tuple[int, int, int, int]
    last_seen: float
    hits: int = 1
    lost_frames: int = 0
    first_seen: float = field(default_factory=time.time)
    in_zone_since: Optional[float] = None  # 进入告警 zone 的时间
    alerted: bool = False  # 是否已告警

    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aa = max(1, (ax2 - ax1) * (ay2 - ay1))
    ba = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(aa + ba - inter + 1e-6)

class VehicleTracker:
    def __init__(self,
                 iou_threshold: float = 0.3,
                 max_lost_frames: int = 5,
                 dwell_threshold_s: float = 30.0,
                 parking_zones: Optional[List[Tuple[int,int,int,int]]] = None):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_id = 1
        self.tracks: List[VehicleTrack] = []
        self.dwell_threshold_s = dwell_threshold_s
        # parking_zones: list of rects (x1,y1,x2,y2)
        self.parking_zones = parking_zones or []
        self.alerts: List[Dict[str,Any]] = []

    def set_parking_zones(self, zones: List[Tuple[int,int,int,int]]):
        self.parking_zones = zones

    def _in_any_zone(self, bbox: Tuple[int,int,int,int]) -> bool:
        # 判断 bbox 中心点是否在任一区域内（简单策略）
        cx, cy = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        for (x1,y1,x2,y2) in self.parking_zones:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True
        return False

    def update(self, detections: List[Dict[str,Any]], ts: Optional[float] = None) -> Tuple[List[VehicleTrack], List[Dict[str,Any]]]:
        """
        detections: list of dict with keys {"bbox":[x1,y1,x2,y2], "confidence":..., "class_id":...}
        返回 (tracks, new_alerts)
        """
        now = ts or time.time()
        det_bboxes = [tuple(map(int, d['bbox'])) for d in detections]

        # 匹配：贪心基于 IoU
        assigned = [-1] * len(det_bboxes)  # index -> track idx
        used_tracks = set()

        for di, db in enumerate(det_bboxes):
            best_i = -1
            best_iou = 0.0
            for ti, tr in enumerate(self.tracks):
                if ti in used_tracks:
                    continue
                score = iou(db, tr.bbox)
                if score > best_iou:
                    best_iou = score
                    best_i = ti
            if best_i != -1 and best_iou >= self.iou_threshold:
                # assign det di -> track best_i
                assigned[di] = best_i
                used_tracks.add(best_i)

        # update assigned tracks
        matched_track_indices = set()
        for di, ti in enumerate(assigned):
            if ti != -1:
                tr = self.tracks[ti]
                tr.bbox = det_bboxes[di]
                tr.last_seen = now
                tr.hits += 1
                tr.lost_frames = 0
                matched_track_indices.add(ti)
                # 处理进入/离开 zone
                if self._in_any_zone(tr.bbox):
                    if tr.in_zone_since is None:
                        tr.in_zone_since = now
                else:
                    tr.in_zone_since = None

        # create new tracks for unmatched detections
        for di, ti in enumerate(assigned):
            if ti == -1:
                nb = det_bboxes[di]
                new_tr = VehicleTrack(track_id=self.next_id, bbox=nb, last_seen=now)
                # set in_zone_since if currently inside
                if self._in_any_zone(nb):
                    new_tr.in_zone_since = now
                self.next_id += 1
                self.tracks.append(new_tr)

        # increase lost_frames for unmatched tracks
        for ti, tr in enumerate(list(self.tracks)):
            if ti not in matched_track_indices:
                tr.lost_frames += 1

        # remove tracks lost too long
        self.tracks = [t for t in self.tracks if t.lost_frames <= self.max_lost_frames]

        # check alerts: any track in zone with dwell >= threshold and not alerted
        new_alerts = []
        for tr in self.tracks:
            if tr.in_zone_since and not tr.alerted:
                dwell = now - tr.in_zone_since
                if dwell >= self.dwell_threshold_s:
                    alert = {
                        "track_id": tr.track_id,
                        "bbox": list(tr.bbox),
                        "dwell_seconds": round(dwell, 1),
                        "timestamp": now,
                        "message": f"车辆在禁停区停留 {int(dwell)} 秒，超出阈值 {int(self.dwell_threshold_s)}s"
                    }
                    tr.alerted = True
                    self.alerts.append(alert)
                    new_alerts.append(alert)

        return list(self.tracks), new_alerts

    def reset(self):
        self.tracks.clear()
        self.next_id = 1
        self.alerts.clear()

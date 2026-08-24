from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class VehicleIdentity:
    stable_id: int
    color: str
    vehicle_type: str
    make: str
    color_conf: float
    type_conf: float
    make_conf: float
    detector_class: str 
    last_bbox: list[float]
    last_center: tuple[float, float]
    last_seen_frame: int
    ema_vector: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    history_centers: list = field(default_factory=list)
    area: float = 0.0
    peak_area: float = 0.0      
    area_history: list = field(default_factory=list) 
    aspect_ratio: float = 0.0
    velocity_mag: float = 0.0
    occluded_by_sid: Optional[int] = None 
    recovery_frames: int = 0 

class VehicleIdentityManager:
    def __init__(self, video_resolution: tuple[int, int], fps: int = 30):
        self.w, self.h = video_resolution
        self.fps = fps
        self.border_margin = int(self.w * 0.03) 
        
        # PHÂN TẦNG DIỆN TÍCH
        self.area_candidate = (self.w * self.h) * 0.0004 
        self.area_register = (self.w * self.h) * 0.0012  
        
        self.birth_persist = 4  
        self.death_grace = int(15 * fps) 
        self.match_score_threshold = 8.0 
        
        self.next_stable_id = 1
        self.active_vehicles: dict[int, VehicleIdentity] = {} 
        self.candidates: dict[int, dict] = {} 
        self.waiting_list: dict[int, VehicleIdentity] = {} 
        self.tracker_to_stable: dict[int, int] = {} 

    def _get_info(self, bbox):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        return ((x1 + x2) / 2, (y1 + y2) / 2), w * h, w / h if h != 0 else 0

    def _calculate_ios(self, target_box, reference_box):
        xA = max(target_box[0], reference_box[0])
        yA = max(target_box[1], reference_box[1])
        xB = min(target_box[2], reference_box[2])
        yB = min(target_box[3], reference_box[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        targetArea = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
        return interArea / float(targetArea + 1e-6)

    def update(self, tracks: list, frame_index: int, attributes_provider, frame):
        active_tids = []
        mature_candidates = []

        for trk in tracks:
            tid, bbox = trk.track_id, trk.bbox
            center, area, ar = self._get_info(bbox)
            
            if tid in self.tracker_to_stable:
                sid = self.tracker_to_stable[tid]
                if sid in self.active_vehicles:
                    self._update_active(sid, bbox, center, area, ar, frame_index)
                    active_tids.append(tid)
                continue

            if tid in self.candidates:
                cand = self.candidates[tid]
                cand['frames'] += 1
                cand['last_seen'] = frame_index
                # Chỉnh sửa: Đạt persist VÀ đạt kích thước register mới đấu thầu
                if cand['frames'] >= self.birth_persist and area >= self.area_register:
                    mature_candidates.append({
                        'tid': tid, 'bbox': bbox, 'center': center, 
                        'area': area, 'ar': ar, 'class': trk.detector_class
                    })
                    del self.candidates[tid] # SỬA LỖI BUG TRÙNG LẶP
                continue

            if area >= self.area_candidate:
                self.candidates[tid] = {'frames': 1, 'last_seen': frame_index}

        if mature_candidates:
            self._process_global_matching(mature_candidates, frame, attributes_provider, frame_index, active_tids)

        self.tracker_to_stable = {k: v for k, v in self.tracker_to_stable.items() if k in active_tids}
        self._cleanup(frame_index)

    def _process_global_matching(self, mature_candidates, frame, attributes_provider, frame_index, active_tids):
        all_possible_bids = []
        for cand in mature_candidates:
            x1, y1, x2, y2 = map(int, np.clip(cand['bbox'], [0, 0, 0, 0], [self.w, self.h, self.w, self.h]))
            if (x2 - x1) <= 5 or (y2 - y1) <= 5: continue
            
            crop = frame[y1:y2, x1:x2]
            attrs = attributes_provider.predict(crop)
            cand['attrs'] = attrs 

            for sid, v in self.waiting_list.items():
                if cand['class'] != v.detector_class: continue
                score = self._calculate_bid_score(cand, v, frame_index)
                if score >= self.match_score_threshold:
                    all_possible_bids.append({'score': score, 'tid': cand['tid'], 'sid': sid, 'cand_data': cand})

        all_possible_bids.sort(key=lambda x: x['score'], reverse=True)
        assigned_tids, assigned_sids = set(), set()

        for bid in all_possible_bids:
            if bid['tid'] in assigned_tids or bid['sid'] in assigned_sids: continue
            
            tid, sid = bid['tid'], bid['sid']
            self.tracker_to_stable[tid] = sid
            v = self.waiting_list.pop(sid)
            v.history_centers = [v.last_center]
            v.recovery_frames = 15
            self.active_vehicles[sid] = v
            self._update_active(sid, bid['cand_data']['bbox'], bid['cand_data']['center'], 
                               bid['cand_data']['area'], bid['cand_data']['ar'], frame_index)
            assigned_tids.add(tid)
            assigned_sids.add(sid)
            active_tids.append(tid)

        for cand in mature_candidates:
            if cand['tid'] not in assigned_tids:
                sid = self.next_stable_id
                self.next_stable_id += 1
                attrs = cand['attrs']
                self.active_vehicles[sid] = VehicleIdentity(
                    stable_id=sid, color=attrs['color'], vehicle_type=attrs['vehicle_type'], make=attrs['make'],
                    color_conf=attrs['color_confidence'], type_conf=attrs['type_confidence'], make_conf=attrs['make_confidence'],
                    last_bbox=cand['bbox'], last_center=cand['center'], last_seen_frame=frame_index, 
                    area=cand['area'], aspect_ratio=cand['ar'], detector_class=cand['class'], peak_area=cand['area']
                )
                self.tracker_to_stable[cand['tid']] = sid
                active_tids.append(cand['tid'])

    def _calculate_bid_score(self, cand, v, frame_index):
        dt = frame_index - v.last_seen_frame
        curr_center = cand['center']
        
        expected_center = np.array(v.last_center) + v.ema_vector * dt
        dist_to_expected = math.dist(curr_center, expected_center)
        max_radius = (v.velocity_mag * dt * 1.5) + (self.w * 0.08)
        motion_score = max(0, 10 * (1 - dist_to_expected / (max_radius + 1e-6)))

        occlusion_bonus = 0
        if v.occluded_by_sid is not None and v.occluded_by_sid in self.active_vehicles:
            blocker = self.active_vehicles[v.occluded_by_sid]
            # Nếu hiện ra gần blocker (IoS > 0)
            if self._calculate_ios(cand['bbox'], blocker.last_bbox) > 0.1:
                occlusion_bonus = 8.0 

        size_score = max(0, 5 * (1 - abs(cand['area'] - v.peak_area) / (v.peak_area + 1e-6)))

        attrs = cand['attrs']
        dna_score = 0
        dna_score += (4 * attrs['color_confidence'] * v.color_conf) if attrs['color'] == v.color else 0
        dna_score += (5 * attrs['type_confidence'] * v.type_conf) if attrs['vehicle_type'] == v.vehicle_type else 0
        dna_score += (4 * attrs['make_confidence'] * v.make_conf) if attrs['make'] == v.make else 0
        
        temporal_penalty = (dt / self.fps) * 0.5 # Trừ 0.5 điểm mỗi giây

        return motion_score + occlusion_bonus + size_score + dna_score - temporal_penalty

    def _cleanup(self, frame_index):
        cands_to_del = [tid for tid, c in self.candidates.items() if (frame_index - c['last_seen']) > 30]
        for tid in cands_to_del: del self.candidates[tid]

        to_deactivate = [sid for sid, v in self.active_vehicles.items() if v.last_seen_frame < frame_index]
        for sid in to_deactivate:
            v = self.active_vehicles.pop(sid)
            x1, y1, x2, y2 = v.last_bbox
            vx, vy = v.ema_vector

            is_leaving = False
            if x1 < self.border_margin and vx < 0: is_leaving = True
            elif x2 > (self.w - self.border_margin) and vx > 0: is_leaving = True
            elif y1 < self.border_margin and vy < 0: is_leaving = True
            elif y2 > (self.h - self.border_margin) and vy > 0: is_leaving = True
            if is_leaving: continue 

            peak_ratio = v.area / (v.peak_area + 1e-6)
            is_tiny = peak_ratio < 0.1
            is_moving_far = False
            if len(v.area_history) >= 3:
                is_moving_far = all(v.area_history[i] > v.area_history[i+1] for i in range(len(v.area_history)-1))
            
            if is_tiny and is_moving_far: continue 

            # Occlusion Context
            v.occluded_by_sid = None
            for other_sid, other_v in self.active_vehicles.items():
                if self._calculate_ios(v.last_bbox, other_v.last_bbox) > 0.15:
                    v.occluded_by_sid = other_sid
                    break
            self.waiting_list[sid] = v
        
        to_purge = [sid for sid, v in self.waiting_list.items() if (frame_index - v.last_seen_frame) > self.death_grace]
        for sid in to_purge: self.waiting_list.pop(sid, None)

    def _update_active(self, sid, bbox, center, area, ar, frame_index):
        v = self.active_vehicles[sid]
        ema_alpha = 0.1 if v.recovery_frames > 0 else 0.3
        if v.recovery_frames > 0: v.recovery_frames -= 1
        v.history_centers.append(center)
        if len(v.history_centers) > 10: v.history_centers.pop(0)
        if len(v.history_centers) >= 2:
            raw = np.array([center[0]-v.history_centers[-2][0], center[1]-v.history_centers[-2][1]])
            v.ema_vector = (1 - ema_alpha) * v.ema_vector + ema_alpha * raw
            v.velocity_mag = np.linalg.norm(v.ema_vector)
        v.last_bbox, v.last_center, v.area, v.aspect_ratio, v.last_seen_frame = bbox, center, area, ar, frame_index
        v.peak_area = max(v.peak_area, area)
        v.area_history.append(area)
        if len(v.area_history) > 5: v.area_history.pop(0)

    def get_display_data(self, tid):
        sid = self.tracker_to_stable.get(tid)
        if sid and sid in self.active_vehicles:
            v = self.active_vehicles[sid]
            status = "recovering" if v.recovery_frames > 0 else "stable"
            return sid, {"color": v.color, "vehicle_type": v.vehicle_type, "make": v.make, "status": status}
        return None, None
import cv2
import numpy as np
import glob
import os

THRESHOLD = 20
MIN_AREA = 2000
DISTANCE_THRESH = 150
MAX_LOST_FRAMES = 20
IOU_THRESHOLD = 0.8
USE_IOU_MATCHING = True
ASPECT_MIN = 0.8
ASPECT_MAX = 5.0

class Track:
    def __init__(self, tid: int, bbox, cx: int, cy: int):
        self.tid = tid
        self.bbox = bbox
        self.cx = cx
        self.cy = cy
        self.lost = 0
        self.age = 1           
        self.history = [(cx, cy)]   

    def update(self, bbox, cx: int, cy: int):
        self.bbox = bbox
        self.cx = cx
        self.cy = cy
        self.lost = 0
        self.age += 1
        self.history.append((cx, cy))
        if len(self.history) > 60:
            self.history.pop(0)

    def mark_lost(self):
        self.lost += 1
        self.age += 1

    @property
    def is_dead(self):
        return self.lost > MAX_LOST_FRAMES

    def predicted_position(self):
        if len(self.history) >= 2:
            dx = self.history[-1][0] - self.history[-2][0]
            dy = self.history[-1][1] - self.history[-2][1]
            return self.cx + dx, self.cy + dy
        return self.cx, self.cy

def update_background(bg: np.ndarray, frame: np.ndarray):
    bg = bg.astype(np.int16)
    frame = frame.astype(np.int16)
    bg[bg < frame] += 1
    bg[bg > frame] -= 1
    return np.clip(bg, 0, 255).astype(np.uint8)

def get_foreground(bg: np.ndarray, frame: np.ndarray):
    diff = cv2.absdiff(frame, bg)
    _, fg = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=4)
    return fg

def get_detections(fg: np.ndarray):
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < MIN_AREA: 
            continue
        cy_check = y + h // 2
        if cy_check < 350: 
            continue
        aspect = h / w
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX): 
            continue
        detections.append((x, y, w, h, x + w // 2, cy_check))
    return detections


def iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
    iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
    inter = ix * iy
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0.0


def centroid_distance(t: Track, cx, cy):
    px, py = t.predicted_position()
    return np.hypot(px - cx, py - cy)


def greedy_match(tracks: dict, detections):
    matched = {}
    used_dets = set()
    track_ids = list(tracks.keys())
    costs = {}
    for tid in track_ids:
        t = tracks[tid]
        for i, (x, y, w, h, cx, cy) in enumerate(detections):
            dist  = centroid_distance(t, cx, cy)
            if USE_IOU_MATCHING:
                iou_score = iou(t.bbox, (x, y, w, h))
                score = dist * (1 - 0.5 * iou_score)
            else:
                score = dist
            costs[(tid, i)] = score
    sorted_pairs = sorted(costs.items(), key=lambda kv: kv[1])
    assigned_tracks = set()
    for (tid, i), score in sorted_pairs:
        if tid in assigned_tracks or i in used_dets:
            continue
        _, _, _, _, cx, cy = detections[i]
        dist = centroid_distance(tracks[tid], cx, cy)
        if dist < DISTANCE_THRESH:
            matched[tid] = i
            assigned_tracks.add(tid)
            used_dets.add(i)
    unmatched_tracks = [tid for tid in track_ids if tid not in assigned_tracks]
    unmatched_dets   = [i for i in range(len(detections)) if i not in used_dets]

    return matched, unmatched_tracks, unmatched_dets
tracks:   dict[int, Track] = {}
next_id:  int = 0

def update_tracks(detections):
    global tracks, next_id
    matched, unmatched_tracks, unmatched_dets = greedy_match(tracks, detections)
    for tid, i in matched.items():
        x, y, w, h, cx, cy = detections[i]
        tracks[tid].update((x, y, w, h), cx, cy)
    for tid in unmatched_tracks:
        tracks[tid].mark_lost()
    tracks = {tid: t for tid, t in tracks.items() if not t.is_dead}
    for i in unmatched_dets:
        x, y, w, h, cx, cy = detections[i]
        tracks[next_id] = Track(next_id, (x, y, w, h), cx, cy)
        next_id += 1

class MOTAEvaluator:
    def __init__(self):
        self.total_gt = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_idsw = 0
        self._prev_assignment: dict = {}

    def update(self, detections, g_t: int):
        n_det = len(detections)
        n_trk = len(tracks)
        fp = max(0, n_trk - n_det)
        fn = max(0, g_t - n_trk)
        idsw = 0
        current_assignment = {}
        for tid, t in tracks.items():
            key = (t.cx // 20, t.cy // 20)
            if key in self._prev_assignment and self._prev_assignment[key] != tid:
                idsw += 1
            current_assignment[key] = tid
        self._prev_assignment = current_assignment
        self.total_gt += g_t
        self.total_fp += fp
        self.total_fn += fn
        self.total_idsw += idsw
        return fp, fn, idsw

    def mota(self) -> float:
        if self.total_gt == 0:
            return 0.0
        return 1.0 - (self.total_fn + self.total_fp + self.total_idsw) / self.total_gt

    def report(self):
        print("\n" + "="*50)
        print("  MOTA EVALUATION REPORT")
        print("="*50)
        print(f"  Total GT objects  (sigmag)   : {self.total_gt}")
        print(f"  False Negatives   (sigmaFN)  : {self.total_fn}")
        print(f"  False Positives   (sigmaFP)  : {self.total_fp}")
        print(f"  Identity Switches (sigmaIDSW): {self.total_idsw}")
        print(f"  MOTA = 1 - (FN+FP+IDSW)/g = {self.mota():.4f}  ({self.mota()*100:.2f}%)")
        print("="*50)

COLOURS = [
    (255,87,34),(33,150,243),(76,175,80),(156,39,176),
    (255,193,7),(0,188,212),(244,67,54),(139,195,74),
]

def colour_for(tid: int):
    return COLOURS[tid % len(COLOURS)]

def draw(frame: np.ndarray, detections, evaluator: MOTAEvaluator):
    for tid, t in tracks.items():
        c = colour_for(tid)
        x, y, w, h = t.bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), c, 2)
        aspect = h / w if w > 0 else 0
        label = f"ID{tid}  ar:{aspect:.1f}"
        cv2.putText(frame, label, (x, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return frame

base_dir = os.path.dirname(os.path.abspath(__file__))
path_pattern = os.path.join(base_dir,r"evs_mot_public_dataset\evs_mot-test\MOT_07\img1\*.jpg")
image_paths  = sorted(glob.glob(path_pattern))
print(f"[INFO] Found {len(image_paths)} frames.")
first = cv2.imread(image_paths[0])
bg = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
evaluator = MOTAEvaluator()

output_txt_path = os.path.join(base_dir, "results.txt")

with open(output_txt_path, "w") as f:
    for frame_idx, path in enumerate(image_paths):
        frame = cv2.imread(path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        bg = update_background(bg, gray)
        fg = get_foreground(bg, gray)
        detections = get_detections(fg)
        update_tracks(detections)
        for tid, t in tracks.items():
            if t.lost == 0:
                x, y, w, h = t.bbox
                line = f"{frame_idx + 1},{tid},{x},{y},{w},{h},1,-1,-1,-1\n"
                f.write(line)
        
        stable_tracks = [t for t in tracks.values() if t.age > 10]
        g_t = len(stable_tracks)
        evaluator.update(detections, g_t)
        
        output = draw(frame.copy(), detections, evaluator)
        cv2.imshow("MOT Tracker", output)
        cv2.imshow("Foreground Mask", fg)
        
        if cv2.waitKey(50) & 0xFF == 27:
            break

cv2.destroyAllWindows()
evaluator.report()
print(f"[INFO] Results saved to: {output_txt_path}")
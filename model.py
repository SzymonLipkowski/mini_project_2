import cv2
import numpy as np
import glob
import os

# ============================================================
# PARAMETERS
# ============================================================
THRESHOLD        = 55          # sigma-delta binarisation threshold (wyższy = mniej szumu/okien)
MIN_AREA         = 1500        # min bounding-box area (pixels²) – odrzuca torby i fragmenty ubrań
DISTANCE_THRESH  = 80          # max centroid distance for assignment (px)
MAX_LOST_FRAMES  = 10          # frames a track survives without a match
IOU_THRESHOLD    = 0.3         # IoU threshold for greedy IoU matching
USE_IOU_MATCHING = True        # True → IoU+centroid hybrid; False → centroid only

# Filtr kształtu – osoba jest wyraźnie wyższa niż szersza
ASPECT_MIN       = 1.3         # min h/w  (odrzuca torby, ręce, poziome obiekty)
ASPECT_MAX       = 4.5         # max h/w  (odrzuca bardzo wąskie pionowe pasy)

# Maska statyczna ROI – lista prostokątów (x, y, w, h) które IGNORUJEMY (np. okna)
# Dodaj/edytuj wpisy jeśli znasz położenie okien w datasecie.
# Przykład: STATIC_IGNORE = [(10, 20, 150, 100), (300, 0, 200, 80)]
STATIC_IGNORE: list[tuple[int,int,int,int]] = []

# ============================================================
# TRACK CLASS  – full lifecycle management
# ============================================================
class Track:
    """Represents a single tracked object with age, lost counter, and history."""

    def __init__(self, tid: int, bbox, cx: int, cy: int):
        self.tid          = tid
        self.bbox         = bbox          # (x, y, w, h)
        self.cx           = cx
        self.cy           = cy
        self.lost         = 0             # consecutive frames without match
        self.age          = 1             # total frames this track has existed
        self.history      = [(cx, cy)]   # for visualisation

    def update(self, bbox, cx: int, cy: int):
        self.bbox    = bbox
        self.cx      = cx
        self.cy      = cy
        self.lost    = 0
        self.age    += 1
        self.history.append((cx, cy))
        if len(self.history) > 60:
            self.history.pop(0)

    def mark_lost(self):
        self.lost += 1
        self.age  += 1

    @property
    def is_dead(self) -> bool:
        return self.lost > MAX_LOST_FRAMES

    def predicted_position(self):
        """Simple linear velocity prediction when lost."""
        if len(self.history) >= 2:
            dx = self.history[-1][0] - self.history[-2][0]
            dy = self.history[-1][1] - self.history[-2][1]
            return self.cx + dx, self.cy + dy
        return self.cx, self.cy


# ============================================================
# SIGMA-DELTA BACKGROUND MODEL
# ============================================================
def update_background(bg: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Sigma-delta adaptive background: increment/decrement pixel-wise."""
    bg = bg.astype(np.int16)
    frame = frame.astype(np.int16)
    bg[bg < frame] += 1
    bg[bg > frame] -= 1
    return np.clip(bg, 0, 255).astype(np.uint8)


# ============================================================
# FOREGROUND EXTRACTION
# ============================================================
def get_foreground(bg: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Robust foreground mask from absolute difference + morphology."""
    diff = cv2.absdiff(frame, bg)
    _, fg = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)

    # 1. Remove tiny noise (trees, leaves)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  np.ones((5,  5),  np.uint8))
    # 2. Bridge gaps between body parts (mniejszy kernel – nie skleja okien z ludźmi)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((25, 10), np.uint8))
    # 3. Fill remaining pin-holes
    fg = cv2.dilate(fg, np.ones((7, 7), np.uint8), iterations=1)

    # 4. Zeruj obszary statyczne (okna, ściany) z listy STATIC_IGNORE
    for (rx, ry, rw, rh) in STATIC_IGNORE:
        fg[ry:ry+rh, rx:rx+rw] = 0

    return fg


# ============================================================
# DETECTION EXTRACTION
# ============================================================
def get_detections(fg: np.ndarray):
    """Return list of (x, y, w, h, cx, cy) for each valid contour."""
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filtr 1: minimalna powierzchnia
        if w * h < MIN_AREA:
            continue

        # Filtr 2: aspect ratio – osoba jest wyraźnie wyższa niż szersza
        aspect = h / w
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            continue

        detections.append((x, y, w, h, x + w // 2, y + h // 2))
    return detections


# ============================================================
# MATCHING HELPERS
# ============================================================
def iou(b1, b2) -> float:
    """Intersection-over-Union for two (x, y, w, h) boxes."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
    iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
    inter = ix * iy
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0.0


def centroid_distance(t: Track, cx, cy) -> float:
    px, py = t.predicted_position()
    return np.hypot(px - cx, py - cy)


def greedy_match(tracks: dict, detections):
    """
    Greedy nearest-neighbour matching combining centroid distance and IoU.
    Returns:
        matched   : {track_id: detection_index}
        unmatched_tracks : list of track_ids
        unmatched_dets   : list of detection indices
    """
    matched           = {}
    used_dets         = set()
    track_ids         = list(tracks.keys())

    # Build cost matrix  (lower = better)
    costs = {}
    for tid in track_ids:
        t = tracks[tid]
        for i, (x, y, w, h, cx, cy) in enumerate(detections):
            dist  = centroid_distance(t, cx, cy)
            if USE_IOU_MATCHING:
                iou_score = iou(t.bbox, (x, y, w, h))
                # blend: distance penalised by lack of overlap
                score = dist * (1 - 0.5 * iou_score)
            else:
                score = dist
            costs[(tid, i)] = score

    # Greedy: assign cheapest pair first
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


# ============================================================
# TRACKER  – global state
# ============================================================
tracks:   dict[int, Track] = {}
next_id:  int = 0


def update_tracks(detections):
    """Full track lifecycle: match → update → mark_lost → kill → create."""
    global tracks, next_id

    matched, unmatched_tracks, unmatched_dets = greedy_match(tracks, detections)

    # Update matched tracks
    for tid, i in matched.items():
        x, y, w, h, cx, cy = detections[i]
        tracks[tid].update((x, y, w, h), cx, cy)

    # Mark unmatched tracks as lost (use velocity prediction)
    for tid in unmatched_tracks:
        tracks[tid].mark_lost()

    # Remove dead tracks
    tracks = {tid: t for tid, t in tracks.items() if not t.is_dead}

    # Spawn new tracks for unmatched detections
    for i in unmatched_dets:
        x, y, w, h, cx, cy = detections[i]
        tracks[next_id] = Track(next_id, (x, y, w, h), cx, cy)
        next_id += 1


# ============================================================
# MOTA METRIC  –  MOTA = 1 − Σ(FN + FP + IDSW) / Σg
# ============================================================
class MOTAEvaluator:
    """
    Online MOTA computation (no ground-truth file needed).
    Counts FP, FN, IDSW relative to the detector's own output so you
    can measure how well the tracker follows detections frame-by-frame.
    When ground-truth is available, replace gt_per_frame accordingly.
    """

    def __init__(self):
        self.total_gt   = 0
        self.total_fp   = 0
        self.total_fn   = 0
        self.total_idsw = 0
        # map detection centroid → last assigned track id  (for IDSW)
        self._prev_assignment: dict = {}

    def update(self, detections, g_t: int):
        """
        detections : list of (x,y,w,h,cx,cy) – what the detector found
        g_t        : ground-truth object count for this frame (use len(detections)
                     as an approximation when no GT file is available)
        """
        n_det = len(detections)
        n_trk = len(tracks)

        # FP: tracker produces more active tracks than detections
        fp = max(0, n_trk - n_det)
        # FN: fewer active tracks than ground-truth objects
        fn = max(0, g_t - n_trk)

        # IDSW: a detection is now matched to a different track than before
        idsw = 0
        current_assignment = {}
        for tid, t in tracks.items():
            key = (t.cx // 20, t.cy // 20)   # coarse grid cell as proxy
            if key in self._prev_assignment and self._prev_assignment[key] != tid:
                idsw += 1
            current_assignment[key] = tid
        self._prev_assignment = current_assignment

        self.total_gt   += g_t
        self.total_fp   += fp
        self.total_fn   += fn
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
        print(f"  Total GT objects  (Σg)   : {self.total_gt}")
        print(f"  False Negatives   (ΣFN)  : {self.total_fn}")
        print(f"  False Positives   (ΣFP)  : {self.total_fp}")
        print(f"  Identity Switches (ΣIDSW): {self.total_idsw}")
        print(f"  MOTA = 1 - (FN+FP+IDSW)/g = {self.mota():.4f}  ({self.mota()*100:.2f}%)")
        print("="*50)


# ============================================================
# VISUALISATION
# ============================================================
COLOURS = [
    (255, 87,  34), (33, 150, 243), (76, 175,  80), (156, 39, 176),
    (255, 193,   7),(0,  188, 212), (244,  67,  54),(139, 195,  74),
]

def colour_for(tid: int):
    return COLOURS[tid % len(COLOURS)]

def draw(frame: np.ndarray, detections, evaluator: MOTAEvaluator) -> np.ndarray:
    # Draw trajectory history
    for tid, t in tracks.items():
        c = colour_for(tid)
        pts = np.array(t.history, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, c, 1)

    # Draw bounding boxes + IDs + aspect ratio
    for tid, t in tracks.items():
        c = colour_for(tid)
        x, y, w, h = t.bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), c, 2)
        aspect = h / w if w > 0 else 0
        label = f"ID{tid}  ar:{aspect:.1f}"
        cv2.putText(frame, label, (x, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)

    # Narysuj obszary ignorowane (STATIC_IGNORE) jako szare prostokąty
    for (rx, ry, rw, rh) in STATIC_IGNORE:
        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (100, 100, 100), 1)
        cv2.putText(frame, "IGNORE", (rx+2, ry+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100,100,100), 1)

    # HUD
    mota_val = evaluator.mota()
    cv2.putText(frame, f"Tracks: {len(tracks)}   MOTA: {mota_val*100:.1f}%",
                (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return frame


# ============================================================
# MAIN – IMAGE SEQUENCE LOOP
# ============================================================
def main():
    base_dir     = os.path.dirname(os.path.abspath(__file__))
    path_pattern = os.path.join(
        base_dir,
        r"evs_mot_public_dataset\evs_mot-test\MOT_01\img1\*.jpg"
    )
    image_paths  = sorted(glob.glob(path_pattern))

    if not image_paths:
        print(f"[ERROR] No images found at:\n  {path_pattern}")
        return

    print(f"[INFO] Found {len(image_paths)} frames.")

    # Initialise background from first frame
    first = cv2.imread(image_paths[0])
    bg    = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    evaluator = MOTAEvaluator()

    for frame_idx, path in enumerate(image_paths):
        frame = cv2.imread(path)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Update sigma-delta background
        bg = update_background(bg, gray)

        # 2. Extract foreground mask
        fg = get_foreground(bg, gray)

        # 3. Detect blobs
        detections = get_detections(fg)

        # 4. Update tracker (full lifecycle)
        update_tracks(detections)

        # 5. Evaluate MOTA for this frame
        #    g_t = number of detections (proxy for GT when no GT file available)
        g_t = len(detections)
        fp, fn, idsw = evaluator.update(detections, g_t)

        # 6. Draw results
        output = draw(frame.copy(), detections, evaluator)

        # Frame counter overlay
        cv2.putText(output, f"Frame {frame_idx+1}/{len(image_paths)}",
                    (6, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.imshow("MOT Tracker", output)
        cv2.imshow("Foreground Mask", fg)

        key = cv2.waitKey(0) & 0xFF   # press any key to advance; ESC to quit
        if key == 27:
            break

    cv2.destroyAllWindows()
    evaluator.report()


if __name__ == "__main__":
    main()
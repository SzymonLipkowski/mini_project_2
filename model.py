import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_BASE_DIR = os.path.join(BASE_DIR, "evs_mot_public_dataset", "evs_mot-test")

OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "submission", "data")

IOU_THRESHOLD = 0.3
MAX_LOST_FRAMES = 10 

class Track:
    def __init__(self, tid, bbox):
        self.tid = tid
        self.bbox = bbox  # [x, y, w, h]
        self.lost = 0
        self.is_active = True

    def update(self, bbox):
        self.bbox = bbox
        self.lost = 0

    def mark_lost(self):
        self.lost += 1
        if self.lost > MAX_LOST_FRAMES:
            self.is_active = False

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

def process_sequence(seq_name):
    
    det_path = os.path.join(INPUT_BASE_DIR, seq_name, "det", "det.txt")
    out_path = os.path.join(OUTPUT_BASE_DIR, f"{seq_name}.txt")
    
    detections = np.loadtxt(det_path, delimiter=',')
    tracks = []
    next_id = 1
    output_lines = []

    frames = np.unique(detections[:, 0]).astype(int)
    
    for frame in range(int(frames.min()), int(frames.max()) + 1):
        current_frame_dets = detections[detections[:, 0] == frame]
        current_bboxes = current_frame_dets[:, 2:6] 
        
        matched_indices = set()
        
        for track in [t for t in tracks if t.is_active]:
            best_iou = -1
            best_det_idx = -1
            
            for i, bbox in enumerate(current_bboxes):
                if i in matched_indices:
                    continue
                
                iou_val = calculate_iou(track.bbox, bbox)
                if iou_val > IOU_THRESHOLD and iou_val > best_iou:
                    best_iou = iou_val
                    best_det_idx = i
            
            if best_det_idx != -1:
                track.update(current_bboxes[best_det_idx])
                matched_indices.add(best_det_idx)
            else:
                track.mark_lost()

        for i, bbox in enumerate(current_bboxes):
            if i not in matched_indices:
                tracks.append(Track(next_id, bbox))
                next_id += 1
        for track in [t for t in tracks if t.is_active and t.lost == 0]:
            x, y, w, h = track.bbox
            line = f"{frame},{track.tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1"
            output_lines.append(line)

    with open(out_path, "w") as f:
        f.write("\n".join(output_lines) + "\n")


sequences = ["MOT_01", "MOT_06", "MOT_07"]

for seq in sequences:
    process_sequence(seq)
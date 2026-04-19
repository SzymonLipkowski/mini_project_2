import numpy as np
import os
import cv2
from scipy.optimize import linear_sum_assignment

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_BASE_DIR = os.path.join(BASE_DIR, "evs_mot_public_dataset", "evs_mot-test")
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "submission", "data")

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

MAX_LOST_FRAMES = 10 
IOU_WEIGHT = 0.5
APPEARANCE_WEIGHT = 0.5
MATCHING_THRESHOLD = 0.6

class Track:
    def __init__(self, tid, bbox, appearance):
        self.tid = tid
        self.bbox = bbox 
        self.appearance = appearance
        self.lost = 0
        self.is_active = True

    def update(self, bbox, appearance):
        self.bbox = bbox
        if self.appearance is not None and appearance is not None:
            self.appearance = 0.8 * self.appearance + 0.2 * appearance 
        else:
            self.appearance = appearance
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
    
    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0

def extract_appearance(frame_img, bbox):
    if frame_img is None:
        return None
        
    x, y, w, h = map(int, bbox)
    img_h, img_w = frame_img.shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, img_w - x), min(h, img_h - y)
    
    patch = frame_img[y:y+h, x:x+w]
    if patch.size == 0:
        return None
        
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_patch], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()

def compare_appearance(hist1, hist2):
    if hist1 is None or hist2 is None:
        return 0.5
    distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return max(0.0, 1.0 - distance)

def process_sequence(seq_name):
    seq_dir = os.path.join(INPUT_BASE_DIR, seq_name)
    det_path = os.path.join(seq_dir, "det", "det.txt")
    img_dir = os.path.join(seq_dir, "img1")
    out_path = os.path.join(OUTPUT_BASE_DIR, f"{seq_name}.txt")
    
    if not os.path.exists(det_path):
        print(f"Skipping {seq_name}: det.txt not found.")
        return

    detections = np.loadtxt(det_path, delimiter=',')
    tracks = []
    next_id = 1
    output_lines = []

    frames = np.unique(detections[:, 0]).astype(int)
    
    for frame in range(int(frames.min()), int(frames.max()) + 1):
        current_frame_dets = detections[detections[:, 0] == frame]
        current_bboxes = current_frame_dets[:, 2:6] 
        

        img_path = os.path.join(img_dir, f"{frame:06d}.jpg")
        frame_img = cv2.imread(img_path) if os.path.exists(img_path) else None
        
        active_tracks = [t for t in tracks if t.is_active]

        current_appearances = [extract_appearance(frame_img, bbox) for bbox in current_bboxes]
        

        num_tracks = len(active_tracks)
        num_dets = len(current_bboxes)
        
        if num_tracks > 0 and num_dets > 0:
            cost_matrix = np.zeros((num_tracks, num_dets))
            
            for t, track in enumerate(active_tracks):
                for d, (bbox, app) in enumerate(zip(current_bboxes, current_appearances)):
                    iou_score = calculate_iou(track.bbox, bbox)
                    app_score = compare_appearance(track.appearance, app)
                    

                    if iou_score == 0:
                        combined_score = 0
                    else:
                        combined_score = (IOU_WEIGHT * iou_score) + (APPEARANCE_WEIGHT * app_score)
                    

                    cost_matrix[t, d] = 1.0 - combined_score
            
            row_inds, col_inds = linear_sum_assignment(cost_matrix)
            
            matched_tracks = set()
            matched_dets = set()
            

            for t_idx, d_idx in zip(row_inds, col_inds):
                score = 1.0 - cost_matrix[t_idx, d_idx]
                if score >= MATCHING_THRESHOLD:
                    active_tracks[t_idx].update(current_bboxes[d_idx], current_appearances[d_idx])
                    matched_tracks.add(t_idx)
                    matched_dets.add(d_idx)
        else:
            matched_tracks = set()
            matched_dets = set()

        for t_idx, track in enumerate(active_tracks):
            if t_idx not in matched_tracks:
                track.mark_lost()

        for d_idx, (bbox, app) in enumerate(zip(current_bboxes, current_appearances)):
            if d_idx not in matched_dets:
                tracks.append(Track(next_id, bbox, app))
                next_id += 1
                
        for track in [t for t in tracks if t.is_active and t.lost == 0]:
            x, y, w, h = track.bbox
            line = f"{frame},{track.tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1"
            output_lines.append(line)

    with open(out_path, "w") as f:
        f.write("\n".join(output_lines) + "\n")
    print(f"Processed {seq_name} successfully.")

sequences = ["MOT_01", "MOT_06", "MOT_07"]

OUTPUT_VIDEO_DIR = os.path.join(BASE_DIR, "visualizations")
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

def get_color_for_id(track_id):
    np.random.seed(int(track_id) * 100)
    return tuple(np.random.randint(0, 255, size=3).tolist())

def create_video_visualization(seq_name):
    print(f"Rozpoczynam generowanie wizualizacji dla: {seq_name}...")
    
    img_dir = os.path.join(INPUT_BASE_DIR, seq_name, "img1")
    txt_path = os.path.join(OUTPUT_BASE_DIR, f"{seq_name}.txt")
    out_video_path = os.path.join(OUTPUT_VIDEO_DIR, f"{seq_name}_tracked.mp4")
    
    if not os.path.exists(txt_path) or not os.path.exists(img_dir):
        print(f"Brak danych dla {seq_name}.")
        return

    results = np.loadtxt(txt_path, delimiter=',')
    images = sorted([img for img in os.listdir(img_dir) if img.endswith(".jpg")])
    if not images: return

    frame_sample = cv2.imread(os.path.join(img_dir, images[0]))
    height, width, layers = frame_sample.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(out_video_path, fourcc, 30, (width, height))

    for frame_idx, img_name in enumerate(images, start=1):
        frame_img = cv2.imread(os.path.join(img_dir, img_name))
        current_frame_objects = results[results[:, 0] == frame_idx]
        
        for obj in current_frame_objects:
            track_id = int(obj[1])
            x, y, w, h = map(int, obj[2:6])
            color = get_color_for_id(track_id)
            
            cv2.rectangle(frame_img, (x, y), (x + w, y + h), color, 2)
            label = f"ID: {track_id}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame_img, (x, y - text_h - 5), (x + text_w, y), color, -1)
            cv2.putText(frame_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        video.write(frame_img)
    video.release()
    print(f"Zapisano wideo: {out_video_path}")

sequences = ["MOT_01", "MOT_06", "MOT_07"]

for seq in sequences:
    process_sequence(seq)
    create_video_visualization(seq)
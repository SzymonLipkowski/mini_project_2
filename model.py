import cv2
import numpy as np
import glob
import os

# -----------------------------
# PARAMETERS
# -----------------------------
threshold = 30
min_area = 400
distance_threshold = 50

# tracking variables (global)
tracks = {}   # id -> (cx, cy)
next_id = 0

# -----------------------------
# SIGMA-DELTA BACKGROUND UPDATE
# -----------------------------
def update_background(bg, frame):
    bg_new = bg.copy()
    bg_new[bg < frame] += 1
    bg_new[bg > frame] -= 1
    return bg_new

# -----------------------------
# FOREGROUND EXTRACTION
# -----------------------------
def get_foreground(bg, frame):
    diff = cv2.absdiff(frame, bg)
    _, fg = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # A very large kernel to bridge the wide gaps you see in your mask
    # (Width 15 to connect arms/legs, Height 40 to connect head/feet)
    big_kernel = np.ones((40, 15), np.uint8)
    
    # 1. CLEAN: Remove the tiny noise in the trees
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    
    # 2. BRIDGE: Use Closing to pull the slivers together into a person shape
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, big_kernel)
    
    # 3. SOLIDIFY: Dilate to fill any remaining tiny pinholes
    fg = cv2.dilate(fg, np.ones((10,10), np.uint8), iterations=1)

    return fg

# -----------------------------
# DETECTIONS
# -----------------------------
def get_detections(fg):
    detections = []
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h > min_area:
            cx = x + w//2
            cy = y + h//2
            detections.append((x, y, w, h, cx, cy))

    return detections

# -----------------------------
# TRACKING (CENTROID MATCHING)
# -----------------------------
def update_tracks(detections):
    global tracks, next_id

    new_tracks = {}
    used = set()

    # match existing tracks
    for tid, (tx, ty) in tracks.items():
        best_dist = 1e9
        best_idx = -1

        for i, (_, _, _, _, cx, cy) in enumerate(detections):
            if i in used:
                continue

            dist = np.hypot(tx - cx, ty - cy)

            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_dist < distance_threshold:
            _, _, _, _, cx, cy = detections[best_idx]
            new_tracks[tid] = (cx, cy)
            used.add(best_idx)

    # create new tracks
    for i, (_, _, _, _, cx, cy) in enumerate(detections):
        if i not in used:
            new_tracks[next_id] = (cx, cy)
            next_id += 1

    tracks = new_tracks

# -----------------------------
# DRAW RESULTS
# -----------------------------
def draw(frame, detections):
    for tid, (cx, cy) in tracks.items():
        cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)
        cv2.putText(frame, f"ID {tid}", (cx, cy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    for (x, y, w, h, _, _) in detections:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

    return frame

# -----------------------------
# MAIN (IMAGE SEQUENCE)
# -----------------------------
base_dir = os.path.dirname(__file__)

# Combine the script location with your relative path
# The 'r' ensures the backslashes are handled correctly
path_pattern = os.path.join(base_dir, r"..\evs_mot_public_dataset\evs_mot-test\MOT_01\img1\*.jpg")

image_paths = sorted(glob.glob(path_pattern))

# initialize background
first = cv2.imread(image_paths[0])
bg = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

for path in image_paths:
    frame = cv2.imread(path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # background update
    bg = update_background(bg, gray)

    # foreground
    fg = get_foreground(bg, gray)

    # detections
    detections = get_detections(fg)

    # tracking
    update_tracks(detections)

    # draw
    output = draw(frame.copy(), detections)

    cv2.imshow("Tracking", output)
    cv2.imshow("Foreground", fg)
    cv2.waitKey(0)
    # if cv2.waitKey(50) & 0xFF == 27:
    #     break

cv2.destroyAllWindows()
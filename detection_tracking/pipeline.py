# pipeline_test.py
import sys
sys.path.append("..")
import os
import cv2
import numpy as np
from siamese_model import FaceDetector
from detection_tracking.tracker import FaceTracker
from utility import align_face  # make sure you have align_face implemented


detector = FaceDetector(device="cpu")  # use "cuda" if GPU available
tracker = FaceTracker()
def get_traces(cap, output_dir):
    frame_idx = 0
    
    print(f"Starting processing... Output: {output_dir}")

    # --- START LOOP ---
    while True:
        ret, frame = cap.read()
        
        # Break loop if video ends
        if not ret:
            print("End of video stream.")
            break

        # 1️⃣ Detect faces
        raw_dets = detector.detect(frame)

        # 2️⃣ Build tracker input (N,5)
        tracker_dets = []
        for det in raw_dets:
            x1, y1, x2, y2 = det["bbox"]
            score = det.get("score", 1.0)
            tracker_dets.append([x1, y1, x2, y2, score])

        tracker_dets = np.array(tracker_dets, dtype=np.float32) if tracker_dets else np.zeros((0,5), dtype=np.float32)

        # 3️⃣ Track faces
        tracks = tracker.update(tracker_dets)

        # 4️⃣ Match tracks to detections using IoU and align faces
        for t in tracks:
            tb = t["bbox"]
            best_det = None
            best_iou = 0.0

            # Find the detection that matches this track
            for det in raw_dets:
                x1, y1, x2, y2 = det["bbox"]
                
                # Calculate IoU
                xi1 = max(tb[0], x1)
                yi1 = max(tb[1], y1)
                xi2 = min(tb[2], x2)
                yi2 = min(tb[3], y2)
                inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                
                area_tb = (tb[2] - tb[0]) * (tb[3] - tb[1])
                area_det = (x2 - x1) * (y2 - y1)
                
                iou = inter / (area_tb + area_det - inter + 1e-6)
                
                if iou > best_iou:
                    best_iou = iou
                    best_det = det

            # If match found and IoU is good
            if best_det is not None and best_iou > 0.5:
                # Align face
                face_img = align_face(frame, best_det["landmarks"])
                if face_img is not None:
                    # Save with frame index and track ID
                    filename = f"frame{frame_idx:04d}_track{t['track_id']}.jpg"
                    cv2.imwrite(os.path.join(output_dir, filename), face_img)

        # Optional: Print progress every 10 frames to reduce clutter
        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx}: detected {len(raw_dets)} faces, tracked {len(tracks)}")
        
        frame_idx += 1
    # --- END LOOP ---

    cap.release()
    print(f"Done! Processed {frame_idx} frames. Faces saved in '{output_dir}'")
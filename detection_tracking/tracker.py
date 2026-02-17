# tracker.py
import numpy as np
import torch
from bytetracker.byte_tracker import BYTETracker

class FaceTracker:
    def __init__(self, fps=30):
        self.tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8, frame_rate=fps)

    def update(self, detections):
        """
        detections: np.ndarray of shape (N,5) -> [x1, y1, x2, y2, score]
        Returns: list of dicts: {'track_id':int, 'bbox':[x1,y1,x2,y2]}
        """
        if detections.shape[0] == 0:
            return []

    
        if detections.shape[1] == 5:
            class_col = np.zeros((detections.shape[0], 1), dtype=np.float32)
            detections = np.hstack([detections, class_col])
        
        # Convert to PyTorch tensor (BYTETracker expects tensors)
        detections = torch.from_numpy(detections)
        
        online_targets = self.tracker.update(detections, None)  # second arg is placeholder

        tracks = []
        # online_targets is a numpy array of shape (N, 7): [x1, y1, x2, y2, track_id, class, score]
        if len(online_targets) > 0:
            for t in online_targets:
                tracks.append({
                    "track_id": int(t[4]),
                    "bbox": [int(t[0]), int(t[1]), int(t[2]), int(t[3])]
                })
        return tracks

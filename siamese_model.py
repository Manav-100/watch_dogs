# detector.py
import numpy as np
from insightface.app.face_analysis import FaceAnalysis

class FaceDetector:
    def __init__(self, device="cuda", embed=False):
        self.embed = embed
        # We explicitly prioritize CUDA and give it the correct device ID
        if device == "cuda":
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0, # Uses your first GPU
                }),
                'CPUExecutionProvider',
            ]
        else:
            providers = ['CPUExecutionProvider']

        # Ensure FaceAnalysis uses these specific providers
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=0 if device=="cuda" else -1, det_size=(640, 640))

    def detect(self, frame):
        """
        Input: BGR frame (OpenCV)
        Output: list of dicts: {'bbox':[x1,y1,x2,y2], 'score':float, 'landmarks':np.ndarray}
        """
        faces = self.app.get(frame)
        if faces:
            faces = [max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))]
        results = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            results.append({
                "bbox": [x1, y1, x2, y2],
                "score": float(face.det_score),
                "landmarks": face.kps.astype(np.float32),
                "face":frame[y1:y2, x1:x2] if not self.embed else None,
                "embedding": face.normed_embedding.astype(np.float32) if self.embed else None
            })
        if results is not None:
            print(f"Detected  face")
        return results
    

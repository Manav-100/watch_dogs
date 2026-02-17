import cv2
import numpy as np
from insightface.app import FaceAnalysis

# 1. Setup
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))

img1_path = r"D:\College\Vscode\watchdogs\test\similarity_test_photos\ben_affleck_1.jpg"
img2_path = r"D:\College\Vscode\watchdogs\test\similarity_test_photos\ben_affleck_2.jpg"

# 2. Get Auto-Computed Embeddings
faces1 = app.get(cv2.imread(img1_path))
faces2 = app.get(cv2.imread(img2_path))

if len(faces1) > 0 and len(faces2) > 0:
    # Get the largest face in each image
    f1 = max(faces1, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    f2 = max(faces2, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))

    # Compare the embeddings ALREADY found by the app
    # (No manual cropping or color flipping)
    emb1 = f1.embedding
    emb2 = f2.embedding
    
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    print("="*40)
    print(f"AUTO-PILOT SIMILARITY: {sim:.4f}")
    print("="*40)
    
    # Visual Check
    # This shows you exactly what the model "saw" internally
    import insightface
    rimg = app.draw_on(cv2.imread(img1_path), faces1)
    cv2.imwrite("debug_auto_draw.jpg", rimg)
else:
    print("Failed to detect faces in one of the images.")
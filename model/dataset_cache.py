import os
import torch
from tqdm import tqdm
from siamese_model import FaceDetector
from model.dataset import PairDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Caching on: {device}")

detector = FaceDetector(device=device, embed=True)

# Point this to your images
dataset = PairDataset(
    parent_dir="D:\\cctv\\watch_dogs\\datasets\\ms1m-arcface",
    encoder=detector.app.models['recognition']
)

# We use batch_size 1 because InsightFace handles the internal batching
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

cached_e1 = []
cached_e2 = []
cached_y = []

print("Extracting embeddings to RAM...")
for e1, e2, y in tqdm(loader, desc="Caching Dataset"):
    cached_e1.append(e1.squeeze(0)) # Remove the DataLoader batch dim
    cached_e2.append(e2.squeeze(0))
    cached_y.append(y.squeeze(0))

# Stack into massive tensors
e1_tensor = torch.stack(cached_e1)
e2_tensor = torch.stack(cached_e2)
y_tensor = torch.stack(cached_y)

cache_path = "D:\\cctv\\watch_dogs\\datasets\\ms1m-arcface\\embedding_cache(3,5).pt"
torch.save({'e1': e1_tensor, 'e2': e2_tensor, 'y': y_tensor}, cache_path)

print(f"Dataset successfully cached to: {cache_path}")
print(f"Total pairs cached: {len(y_tensor)}")
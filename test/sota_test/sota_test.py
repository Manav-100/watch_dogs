import os
import sys
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

# ==========================================
# 0. STRICT LOCAL PATHING
# ==========================================
# We define CURRENT_DIR as the only source of truth for files
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# We still need the PROJECT_ROOT to find siamese_model.py
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from siamese_model import FaceDetector
    # SOTA models are inside the local folder
    from model_architectures.ada_face.AdaFace import net as adaface_net
    from model_architectures.mag_face.MagFace.models import iresnet as magface_net
except ImportError as e:
    print(f"IMPORT ERROR: {e}\nCheck that siamese_model.py is at {PROJECT_ROOT}")
    sys.exit(1)

# ==========================================
# 1. ARCHITECTURE (v2.16)
# ==========================================
class FaceEncoder(nn.Module):
    def __init__(self, onnx_model, device):
        super().__init__()
        self.model = onnx_model
        self.device = device

    def forward(self, x):
        # Move image to CPU/Numpy for InsightFace
        x_np = x.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        
        embeddings = []
        for i in range(x_np.shape[0]):
            feat = self.model.get_feat(x_np[i])
            embeddings.append(feat)
            
        # STACK AND RESHAPE: Force to [Batch, 512]
        # This prevents the (32, 1, 512) issue causing the 64x512 error
        out = torch.from_numpy(np.stack(embeddings)).float()
        return out.reshape(x.shape[0], 512).to(self.device)

class InternalAttentionHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, e1, e2):
        # Combined shape must be [Batch, 1024]
        combined = torch.cat([e1, e2], dim=1) 
        
        weights = self.attn(combined)
        diff = torch.abs(e1 - e2)
        
        # Classifier gets [Batch, 512]
        return self.classifier(diff * weights).squeeze(1)

class Watch_Dogs(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder, self.head = encoder, head
        
    def forward(self, img1, img2):
        e1 = self.encoder(img1) # Shape: [B, 512]
        e2 = self.encoder(img2) # Shape: [B, 512]
        
        # Standardize normalization
        e1 = F.normalize(e1, p=2, dim=1)
        e2 = F.normalize(e2, p=2, dim=1)
        
        return self.head(e1, e2)
# ==========================================
# 2. DATASET LOADER (STRICTLY LOCAL)
# ==========================================
class TinyFaceDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.pairs, self.path_map = [], {}
        # Looking for 'tinyface' folder exactly where the script is
        dataset_root = os.path.join(CURRENT_DIR, "tinyface")
        
        print(f"Indexing local images in: {dataset_root}")
        for root, _, files in os.walk(dataset_root):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.path_map[f.lower()] = os.path.join(root, f)
        
        if not self.path_map:
            print(f"CRITICAL: No images found in {dataset_root}")
            sys.exit(1)

        print(f"Found {len(self.path_map)} images.")

        with open(csv_path, 'r') as f:
            for line in f:
                p1, p2, l = line.strip().split(',')
                n1, n2 = os.path.basename(p1).lower(), os.path.basename(p2).lower()
                if n1 in self.path_map and n2 in self.path_map:
                    self.pairs.append((n1, n2, int(l)))

        print(f"Success: {len(self.pairs)} valid pairs for evaluation.")
        self.transform = transform or transforms.Compose([
            transforms.Resize((112, 112)), transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255.0)
        ])

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        n1, n2, l = self.pairs[idx]
        return self.transform(Image.open(self.path_map[n1]).convert('RGB')), \
               self.transform(Image.open(self.path_map[n2]).convert('RGB')), \
               torch.tensor(l, dtype=torch.float32)

# ==========================================
# 3. EVALUATION TOOLS
# ==========================================
class FaceModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
        self.norm = transforms.Normalize(mean=[127.5]*3, std=[127.5]*3)

class WatchDogsWrapper(FaceModelWrapper):
    def forward(self, i1, i2): return self.model(i1, i2)

class SOTAWrapper(FaceModelWrapper):
    def forward(self, i1, i2):
        i1, i2 = self.norm(i1/255.0), self.norm(i2/255.0)
        
        # Get raw outputs
        out1 = self.model(i1)
        out2 = self.model(i2)
        
        # Flexibly extract embeddings
        # If output is a tuple (like AdaFace), take the first element.
        # If it's just a tensor (like MagFace), use it directly.
        e1 = out1[0] if isinstance(out1, (tuple, list)) else out1
        e2 = out2[0] if isinstance(out2, (tuple, list)) else out2
        
        return F.cosine_similarity(e1, e2, dim=1)

class BenchmarkEngine:
    def __init__(self, loader, device):
        self.loader, self.device = loader, device
    def evaluate(self, wrapper, name):
        wrapper.to(self.device)
        scores, labels, times = [], [], []
        print(f"\n[Benchmarking] {name}...")
        with torch.no_grad():
            for i1, i2, l in tqdm(self.loader):
                i1, i2 = i1.to(self.device), i2.to(self.device)
                t0 = time.perf_counter()
                s = wrapper(i1, i2)
                times.append((time.perf_counter()-t0)/2.0)
                scores.extend(s.cpu().numpy()); labels.extend(l.numpy())
        fpr, tpr, _ = roc_curve(labels, scores)
        return {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr), "fps": 1.0/np.mean(times)}

def load_local_weights(model, path, device):
    print(f"Loading weights: {path}")
    sd = torch.load(path, map_location=device, weights_only=False)
    sd = sd.get('state_dict', sd.get('model_state_dict', sd))
    sd = {k.replace('model.', '').replace('backbone.', '').replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model.eval()

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # EVERYTHING IS RELATIVE TO CURRENT_DIR
    files = {
        "csv": os.path.join(CURRENT_DIR, "tinyface_pairs.csv"),
        "ada": os.path.join(CURRENT_DIR, "model_weights", "ada_face.ckpt"),
        "mag": os.path.join(CURRENT_DIR, "model_weights", "mag_face.pth"),
        "aro": os.path.join(CURRENT_DIR, "model_weights", "aro_face.pt"),
        "wd": os.path.join(PROJECT_ROOT, "trained", "bollywood_faces", "v1(3,5)", "epoch_19.pth")
    }

    # Ensure files exist before starting
    for k, v in files.items():
        if not os.path.exists(v):
            print(f"FILE NOT FOUND: {v}"); sys.exit(1)

    loader = DataLoader(TinyFaceDataset(files["csv"]), batch_size=64, shuffle=False)
    engine = BenchmarkEngine(loader, device)

    # 1. Initialize Standard Models
    ada = load_local_weights(adaface_net.build_model('ir_50'), files["ada"], device)
    mag = load_local_weights(magface_net.iresnet50(), files["mag"], device)
    aro = load_local_weights(adaface_net.build_model('ir_50'), files["aro"], device)

    # 2. Initialize Watch Dogs
    print("\n[Initializing Watch Dogs System...]")
    det = FaceDetector(device=device, embed=True)
    head = InternalAttentionHead()
    head.load_state_dict(torch.load(files["wd"], map_location=device, weights_only=False))
    encoder = FaceEncoder(det.app.models['recognition'], device) 
    wd = Watch_Dogs(encoder, head).to(device).eval()

    # 3. Run Benchmark
    models = {
        "Watch Dogs (Ours)": WatchDogsWrapper(wd),

        "ARoFace": SOTAWrapper(aro),

        "MagFace": SOTAWrapper(mag),

        "AdaFace": SOTAWrapper(ada)
    }

    results = {n: engine.evaluate(m, n) for n, m in models.items()}

    # --- PLOTTING ---
    plt.figure(figsize=(10, 8))
    for n, r in results.items():
        plt.plot(r["fpr"], r["tpr"], label=f'{n} (AUC={r["auc"]:.3f}, {r["fps"]:.1f} FPS)')
    
    plt.xscale('log'); plt.xlim([1e-4, 1]); plt.ylim([0, 1])
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel('FAR'); plt.ylabel('TAR'); plt.title('SOTA Comparison on QMUL-TinyFace')
    plt.legend(); plt.savefig('benchmark_plot_v1(3,5).png'); plt.show()
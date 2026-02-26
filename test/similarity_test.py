import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import re

from siamese_model import FaceDetector
from custom_model import FaceEncoder
from insightface.utils import face_align 
from model.distortion_simulation import custom_distortion_pipeline

# --- CONFIGURATION ---
IMG_1_PATH = r"D:\cctv\watch_dogs\test\similarity_test_photos\neha_1.jpg"
IMG_2_PATH = r"D:\cctv\watch_dogs\test\similarity_test_photos\neha_2.jpg"

CHECKPOINTS = {
    "Model_V1": r"trained/bollywood_faces/v2.16/epoch_3.pth",
    "Model_V2": r"trained/bollywood_faces/v2.32_margin/epoch_3.pth" 
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. DYNAMIC MODEL ARCHITECTURE ---
class DynamicAttentionHead(nn.Module):
    def __init__(self, attn_modules, clf_modules):
        super().__init__()
        self.attn = nn.Sequential(*attn_modules)
        self.classifier = nn.Sequential(*clf_modules)

    def forward(self, e1, e2):
        # We perform the forward pass steps manually in the loop below 
        # for debugging purposes, but this is the standard logic:
        diff = torch.abs(e1 - e2)
        concat = torch.cat([e1, e2], dim=1)
        weights = self.attn(concat)
        weighted = diff * weights
        return self.classifier(weighted).squeeze(1)

# --- 2. FACTORY: Build Model from Checkpoint ---
def build_model_from_checkpoint(checkpoint_path):
    print(f"[{os.path.basename(checkpoint_path)}] Analyzing...")
    try:
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    except Exception as e:
        print(f"   -> Error loading file: {e}")
        return None

    # A. Clean Keys (Remove 'head.' prefix)
    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("head."):
            clean_state[k.replace("head.", "")] = v
        elif "encoder" not in k:
            clean_state[k] = v
            
    # B. Identify Layer Structure via Regex
    attn_indices = [int(x) for x in re.findall(r'attn\.(\d+)\.weight', str(clean_state.keys()))]
    clf_indices = [int(x) for x in re.findall(r'classifier\.(\d+)\.weight', str(clean_state.keys()))]
    
    if not attn_indices or not clf_indices:
        print("   -> Failed to identify layer structure.")
        return None

    max_attn = max(attn_indices)
    max_clf = max(clf_indices)

    # C. Reconstruct 'attn' Block
    attn_layers = []
    for i in range(0, max_attn + 1):
        w_key = f"attn.{i}.weight"
        b_key = f"attn.{i}.bias"
        
        if w_key in clean_state: # Linear Layer
            weight = clean_state[w_key]
            out_f, in_f = weight.shape
            layer = nn.Linear(in_f, out_f)
            layer.weight.data = weight
            if b_key in clean_state:
                layer.bias.data = clean_state[b_key]
            attn_layers.append(layer)
        elif i == max_attn and "Sigmoid" not in str(attn_layers[-1]):
             # End of block usually has Sigmoid
             pass 
        elif i % 2 != 0: 
             # Odd indices are usually ReLUs
             attn_layers.append(nn.ReLU())

    # Force Final Sigmoid for Attn
    if not isinstance(attn_layers[-1], nn.Sigmoid):
        attn_layers.append(nn.Sigmoid())

    # D. Reconstruct 'classifier' Block
    clf_layers = []
    for i in range(0, max_clf + 1):
        w_key = f"classifier.{i}.weight"
        b_key = f"classifier.{i}.bias"
        
        if w_key in clean_state:
            weight = clean_state[w_key]
            out_f, in_f = weight.shape
            layer = nn.Linear(in_f, out_f)
            layer.weight.data = weight
            if b_key in clean_state:
                layer.bias.data = clean_state[b_key]
            clf_layers.append(layer)
        elif i % 2 != 0:
            clf_layers.append(nn.ReLU())

    # E. Build & Return
    model = DynamicAttentionHead(attn_layers, clf_layers).to(DEVICE)
    model.eval()
    
    # Debug: Print first few layers to confirm construction
    print(f"   -> Built: {len(attn_layers)} Attn Layers / {len(clf_layers)} Clf Layers")
    return model

# --- SETUP ---
print("\nInitializing Encoders...")
detector_detect = FaceDetector(device=DEVICE)
onnx_encoder = FaceDetector(device=DEVICE, embed=True).app.models['recognition']
base_encoder = FaceEncoder(onnx_encoder)

def get_aligned_face(path):
    img = cv2.imread(path)
    if img is None: return None
    results = detector_detect.detect(img)
    if not results: return None
    return face_align.norm_crop(img, landmark=results[0]["landmarks"])

def to_tensor(img_bgr):
    img_uint8 = img_bgr.astype(np.uint8)
    return torch.from_numpy(img_uint8).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

# --- EXECUTION ---
print("Processing Images...")
face1 = get_aligned_face(IMG_1_PATH)
#face2 = get_aligned_face(IMG_2_PATH)
face2=cv2.imread(IMG_2_PATH)

if face1 is None or face2 is None:
    exit("Detection failed for one or more images.")

# Optional: Distortion
dist_res = custom_distortion_pipeline(image=face2)
face2_dist = dist_res["image"]
t1 = to_tensor(face1)
#t2 = to_tensor(face2_dist)
t2=to_tensor(face2)

# --- COMPARISON TABLE ---
print("\n" + "="*95)
print(f"{'Model':<12} | {'Attn Mean':<10} | {'Attn Max':<10} | {'Raw Cos':<8} | {'Prob %':<8} | {'Verdict'}")
print("-" * 95)

with torch.no_grad():
    # Pre-calculate embeddings once
    raw_e1 = base_encoder(t1)
    raw_e2 = base_encoder(t2)
    
    # 1. Ensure they are tensors (just in case ONNX returned raw numpy arrays)
    if not isinstance(raw_e1, torch.Tensor): raw_e1 = torch.tensor(raw_e1)
    if not isinstance(raw_e2, torch.Tensor): raw_e2 = torch.tensor(raw_e2)
    
    # 2. Push them to the GPU
    raw_e1 = raw_e1.to(DEVICE)
    raw_e2 = raw_e2.to(DEVICE)
    
    if raw_e1.dim() == 1: raw_e1 = raw_e1.unsqueeze(0)
    if raw_e2.dim() == 1: raw_e2 = raw_e2.unsqueeze(0)
    
    # Normalize (CRITICAL)
    e1_norm = F.normalize(raw_e1, p=2, dim=1)
    e2_norm = F.normalize(raw_e2, p=2, dim=1)
    cosine = F.cosine_similarity(e1_norm, e2_norm).item()

    for name, path in CHECKPOINTS.items():
        model = build_model_from_checkpoint(path)
        
        if model:
            # --- MANUAL FORWARD PASS (To Inspect Internals) ---
            diff = torch.abs(e1_norm - e2_norm)
            concat = torch.cat([e1_norm, e2_norm], dim=1)
            
            # 1. Inspect Attention
            attn_weights = model.attn(concat)
            attn_mean = attn_weights.mean().item()
            attn_max = attn_weights.max().item()
            
            # 2. Finish Calculation
            weighted = diff * attn_weights
            logit = model.classifier(weighted).squeeze(1)
            prob = torch.sigmoid(logit).item()
            
            verdict = "MATCH" if prob > 0.5 else "NO MATCH"
            
            print(f"{name:<12} | {attn_mean:.4f}     | {attn_max:.4f}     | {cosine:.4f}   | {prob:.4f}   | {verdict}")
        else:
            print(f"{name:<12} | {'ERROR':<10} | {'ERROR':<10} | {cosine:.4f}   | {'ERROR':<8} | ERROR")
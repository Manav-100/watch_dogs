import sys
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ==========================================
# 1. SYSTEM SETUP & PATH INJECTION
# ==========================================
REPO_DIR = "D:\\cctv\\watch_dogs"
if REPO_DIR not in sys.path:
    sys.path.append(REPO_DIR)

try:
    from model.dataset import PairDataset
    from siamese_model import FaceDetector
    print("[SETUP] Core modules imported successfully.")
except ImportError as e:
    print(f"[CRITICAL ERROR] Failed to import modules. Ensure your repo is at {REPO_DIR}\nError: {e}")
    sys.exit(1)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ==========================================
# 2. INLINE MODEL DEFINITION (2-EXIT CASCADE)
# ==========================================
class AttentionHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        # Concatenated input is 1024 (dim * 2)

        # 1. THE HARD EXIT (Evaluated First, Massive Capacity)
        self.layer1_hard = nn.Linear(dim * 2, 512)
        self.hard_classifier = nn.Sequential(
            nn.ReLU(), nn.Linear(512, 128),
            nn.ReLU(), nn.Linear(128, 1)
        )

        # 2. THE EASY EXIT (Fallback, Low Capacity)
        self.layer2_easy = nn.Linear(512, 32)
        self.easy_classifier = nn.Sequential(
            nn.ReLU(), nn.Linear(32, 16),
            nn.ReLU(), nn.Linear(16, 1)
        )

    # Threshold set to 99.9% targeting the 10^-3 to 10^-4 FAR range
    def forward(self, e1, e2, thresholds=0.999, return_all=False):
        concat = torch.cat([e1, e2], dim=1) 

        # Pass 1: Hard
        x_hard = self.layer1_hard(concat)
        logits_hard = self.hard_classifier(x_hard)

        # Pass 2: Easy
        x_easy = self.layer2_easy(F.relu(x_hard))
        logits_easy = self.easy_classifier(x_easy)

        if self.training or return_all:
            return logits_hard, logits_easy

        # --- INFERENCE ROUTING (Probability Based) ---
        prob_hard = torch.sigmoid(logits_hard)
        conf_hard = torch.max(prob_hard, 1.0 - prob_hard)
        
        # Routing Logic
        use_hard = (conf_hard >= thresholds) 
        use_easy = (~use_hard) 

        # Select the winning logit
        final_logits = torch.zeros_like(logits_hard)
        final_logits[use_hard] = logits_hard[use_hard]
        final_logits[use_easy] = logits_easy[use_easy]

        return final_logits

# ==========================================
# 3. STANDARD CHECKPOINT LOADER
# ==========================================
def load_cascade_model(checkpoint_path, model, device):
    print(f"\n[LOADER] Attempting to load: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"[LOADER] Checkpoint file not found. Starting fresh.")
        return model

    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        clean_state = {k.replace("head.", "").replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state, strict=False)
        print("[LOADER] SUCCESS: Cascade model weights loaded.")
    except Exception as e:
        print(f"[LOADER] CRITICAL ERROR during loading: {e}. Starting fresh.")
    
    return model

# ==========================================
# 4. TRAINING CONFIGURATION & DATA SPLIT
# ==========================================
pwd = os.getcwd()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

checkpoint_dir = os.path.join(REPO_DIR, "trained", "bollywood_faces", "v2.32_with_512_margin_exit")
os.makedirs(checkpoint_dir, exist_ok=True)

# User-requested checkpoint
RESUME_CHECKPOINT = r"D:\cctv\watch_dogs\trained\bollywood_faces\v2.32_with_512_margin_exit\epoch_3.pth"

print("[INIT] Loading Cached Embeddings directly to RAM...")
cache_path = "D:\\cctv\\watch_dogs\\datasets\\ms1m-arcface\\embedding_cache(3,5).pt"

if not os.path.exists(cache_path):
    print(f"[CRITICAL ERROR] Cache not found at {cache_path}. Run cache_dataset.py first!")
    sys.exit(1)

cached_data = torch.load(cache_path)
full_dataset = torch.utils.data.TensorDataset(
    cached_data['e1'], 
    cached_data['e2'], 
    cached_data['y']
)

# Split dataset into 90% Training, 10% Validation
val_size = int(len(full_dataset) * 0.1)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Restored batch size to 1024
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

print(f"[INIT] Total Pairs: {len(full_dataset)} | Training: {train_size} | Validation: {val_size}")

model = AttentionHead(dim=512).to(device)

if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
    model = load_cascade_model(RESUME_CHECKPOINT, model, device)
else:
    print("[WARNING] RESUME_CHECKPOINT not found. Starting with fresh weights!")

criterion = nn.BCEWithLogitsLoss()
# Restored LR to 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

scaler = torch.amp.GradScaler('cuda')

log_file = os.path.join(checkpoint_dir, "training_history.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s | %(message)s')
logging.info(f"--- Started Training 2-Exit Cascade Watch Dogs on {device} ---")

# ==========================================
# 5. THE TRAINING & VALIDATION LOOP
# ==========================================
# Restored EPOCHS to 60
EPOCHS = 60

# Restored Margins
MARGIN_HARD = 0.35
MARGIN_EASY = 0.000

# Restored loop starting at epoch 3
for epoch in range(4, EPOCHS):
    # --- TRAINING PHASE ---
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", unit="batch")

    for e1, e2, y in pbar:
        e1 = e1.to(device, non_blocking=True)
        e2 = e2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        e1 = F.normalize(e1, p=2, dim=1)
        e2 = F.normalize(e2, p=2, dim=1)

        targets = y.float().view(-1)
        target_signs = (2.0 * targets - 1.0)

        with torch.amp.autocast('cuda'):
            logits_hard, logits_easy = model(e1, e2)

            adj_hard = logits_hard.view(-1) - (MARGIN_HARD * target_signs)
            adj_easy = logits_easy.view(-1) - (MARGIN_EASY * target_signs)

            loss_hard = criterion(adj_hard, targets)
            loss_easy = criterion(adj_easy, targets)

            loss = loss_hard + loss_easy

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        current_loss = loss.item()
        total_loss += current_loss
        pbar.set_postfix(loss=f"{current_loss:.4f}")

    avg_train_loss = total_loss / len(train_loader)

    # --- VALIDATION PHASE ---
    model.eval()
    correct_hard, correct_easy = 0, 0
    total_val = 0
    
    print(f"\n[Val] Running validation check on {val_size} pairs...")
    with torch.no_grad():
        for e1, e2, y in val_loader:
            e1 = e1.to(device, non_blocking=True)
            e2 = e2.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            e1 = F.normalize(e1, p=2, dim=1)
            e2 = F.normalize(e2, p=2, dim=1)
            
            logits_hard, logits_easy = model(e1, e2, return_all=True)
            targets = y.float().view(-1)

            preds_hard = (torch.sigmoid(logits_hard.view(-1)) >= 0.5).float()
            preds_easy = (torch.sigmoid(logits_easy.view(-1)) >= 0.5).float()

            correct_hard += (preds_hard == targets).sum().item()
            correct_easy += (preds_easy == targets).sum().item()
            total_val += targets.size(0)

    acc_hard = (correct_hard / total_val) * 100
    acc_easy = (correct_easy / total_val) * 100

    summary_msg = (
        f"Epoch {epoch+1}/{EPOCHS} Summary | Train Loss: {avg_train_loss:.4f} | "
        f"Hard Acc: {acc_hard:.2f}% | Easy Acc: {acc_easy:.2f}%"
    )
    print(f"=== {summary_msg} ===\n")
    logging.info(summary_msg)

    save_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_path)

print("Training Complete.")
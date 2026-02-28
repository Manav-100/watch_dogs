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

# ==========================================
# 2. INLINE MODEL DEFINITION (MULTI-EXIT)
# ==========================================
class AttentionHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        # Concatenated input is 1024 (dim * 2)

        # 1. THE HARD EXIT
        self.layer1_hard = nn.Linear(dim * 2, 512)
        self.hard_classifier = nn.Sequential(
            nn.ReLU(), nn.Linear(512, 128),
            nn.ReLU(), nn.Linear(128, 1)
        )

        # 2. THE MEDIUM EXIT
        self.layer2_medium = nn.Linear(512, 128)
        self.medium_classifier = nn.Sequential(
            nn.ReLU(), nn.Linear(128, 64),
            nn.ReLU(), nn.Linear(64, 1)
        )

        # 3. THE EASY EXIT
        self.layer3_easy = nn.Linear(128, 32)
        self.easy_classifier = nn.Sequential(
            nn.ReLU(), nn.Linear(32, 16),
            nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, e1, e2, confidence_threshold=1.5, return_all=False):
        concat = torch.cat([e1, e2], dim=1)

        x_hard = self.layer1_hard(concat)
        logits_hard = self.hard_classifier(x_hard)

        x_medium = self.layer2_medium(F.relu(x_hard))
        logits_medium = self.medium_classifier(x_medium)

        x_easy = self.layer3_easy(F.relu(x_medium))
        logits_easy = self.easy_classifier(x_easy)

        # Return all logits if training OR if explicitly requested for validation scoring
        if self.training or return_all:
            return logits_hard, logits_medium, logits_easy

        conf_hard = torch.abs(logits_hard)
        conf_medium = torch.abs(logits_medium)

        use_hard = (conf_hard >= confidence_threshold)
        use_medium = (~use_hard) & (conf_medium >= confidence_threshold)
        use_easy = (~use_hard) & (~use_medium)

        final_logits = torch.zeros_like(logits_hard)
        final_logits[use_hard] = logits_hard[use_hard]
        final_logits[use_medium] = logits_medium[use_medium]
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

RESUME_CHECKPOINT = "D:\\cctv\\watch_dogs\\trained\\bollywood_faces\\v2.32_margin_exit\\epoch_3.pth" 

print("[INIT] Loading Face Detector...")
detector = FaceDetector(device=device, embed=True)

print("[INIT] Loading Dataset...")
full_dataset = PairDataset(
    parent_dir="D:\\cctv\\watch_dogs\\ms1m-arcface",
    encoder=detector.app.models['recognition']
)

# Split dataset into 90% Training, 10% Validation
val_size = int(len(full_dataset) * 0.1)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print(f"[INIT] Total Pairs: {len(full_dataset)} | Training: {train_size} | Validation: {val_size}")

model = AttentionHead(dim=512).to(device)

if RESUME_CHECKPOINT:
    model = load_cascade_model(RESUME_CHECKPOINT, model, device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

checkpoint_dir = os.path.join(pwd, "trained", "bollywood_faces", "v2.32_margin_exit")
os.makedirs(checkpoint_dir, exist_ok=True)

log_file = os.path.join(checkpoint_dir, "training_history.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s | %(message)s')
logging.info(f"--- Started Training Cascade Watch Dogs on {device} ---")

# ==========================================
# 5. THE TRAINING & VALIDATION LOOP
# ==========================================
EPOCHS = 24

# Staggered Margins
MARGIN_HARD = 0.225
MARGIN_MEDIUM = 0.100
MARGIN_EASY = 0.000

for epoch in range(3,EPOCHS):
    # --- TRAINING PHASE ---
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", unit="batch")

    for e1, e2, y in pbar:
        e1, e2, y = e1.to(device), e2.to(device), y.to(device)
        e1 = F.normalize(e1, p=2, dim=1)
        e2 = F.normalize(e2, p=2, dim=1)

        logits_hard, logits_medium, logits_easy = model(e1, e2)
        targets = y.float().view(-1)

        target_signs = (2.0 * targets - 1.0)

        adj_hard = logits_hard.view(-1) - (MARGIN_HARD * target_signs)
        adj_medium = logits_medium.view(-1) - (MARGIN_MEDIUM * target_signs)
        adj_easy = logits_easy.view(-1) - (MARGIN_EASY * target_signs)

        loss_hard = criterion(adj_hard, targets)
        loss_medium = criterion(adj_medium, targets)
        loss_easy = criterion(adj_easy, targets)

        loss = loss_hard + loss_medium + loss_easy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        total_loss += current_loss
        pbar.set_postfix(loss=f"{current_loss:.4f}")

    avg_train_loss = total_loss / len(train_loader)

    # --- VALIDATION PHASE ---
    model.eval()
    correct_hard, correct_medium, correct_easy = 0, 0, 0
    total_val = 0
    
    print(f"\n[Val] Running validation check on {val_size} pairs...")
    with torch.no_grad():
        for e1, e2, y in val_loader:
            e1, e2, y = e1.to(device), e2.to(device), y.to(device)
            e1 = F.normalize(e1, p=2, dim=1)
            e2 = F.normalize(e2, p=2, dim=1)
            
            # Use return_all=True to inspect all exits simultaneously
            logits_hard, logits_medium, logits_easy = model(e1, e2, return_all=True)
            targets = y.float().view(-1)

            # Convert logits to 0/1 predictions
            preds_hard = (torch.sigmoid(logits_hard.view(-1)) >= 0.5).float()
            preds_medium = (torch.sigmoid(logits_medium.view(-1)) >= 0.5).float()
            preds_easy = (torch.sigmoid(logits_easy.view(-1)) >= 0.5).float()

            correct_hard += (preds_hard == targets).sum().item()
            correct_medium += (preds_medium == targets).sum().item()
            correct_easy += (preds_easy == targets).sum().item()
            total_val += targets.size(0)

    # Calculate independent accuracy for each exit
    acc_hard = (correct_hard / total_val) * 100
    acc_medium = (correct_medium / total_val) * 100
    acc_easy = (correct_easy / total_val) * 100

    summary_msg = (
        f"Epoch {epoch+1}/{EPOCHS} Summary | Train Loss: {avg_train_loss:.4f} | "
        f"Hard Acc: {acc_hard:.2f}% | Med Acc: {acc_medium:.2f}% | Easy Acc: {acc_easy:.2f}%"
    )
    print(f"=== {summary_msg} ===\n")
    logging.info(summary_msg)

    save_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_path)

print("Training Complete.")
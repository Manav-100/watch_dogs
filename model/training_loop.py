import sys
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
    print(f"[CRITICAL ERROR] Failed to import modules. Ensure you are running this from your repo root.\nError: {e}")
    sys.exit(1)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ==========================================
# 2. 2-EXIT INVERTED CASCADE ARCHITECTURE
# ==========================================
class AttentionHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        
        # 1. THE HARD EXIT (Evaluated First, Massive Capacity)
        self.layer1_hard = nn.Linear(dim * 2, 512)
        self.hard_classifier = nn.Sequential(
            nn.ReLU(), nn.Linear(512, 128),
            nn.ReLU(), nn.Linear(128, 1)
        )

        # 2. THE EASY EXIT (Fallback, Low Capacity)
        # Slices the 512 down to 32
        self.layer2_easy = nn.Linear(512, 32)
        self.easy_classifier = nn.Sequential(
            nn.ReLU(), nn.Linear(32, 16),
            nn.ReLU(), nn.Linear(16, 1)
        )

    # Threshold set to 99.9% to target the 10^-3 to 10^-4 FAR range
    def forward(self, e1, e2, threshold=0.999, return_all=False):
        concat = torch.cat([e1, e2], dim=1) 

        # Pass 1: Hard
        x_hard = self.layer1_hard(concat)
        logits_hard = self.hard_classifier(x_hard)

        # Pass 2: Easy
        x_easy = self.layer2_easy(F.relu(x_hard))
        logits_easy = self.easy_classifier(x_easy)

        if self.training or return_all:
            return logits_hard, logits_easy

        # --- INFERENCE ROUTING ---
        prob_hard = torch.sigmoid(logits_hard)
        
        # Calculate confidence
        conf_hard = torch.max(prob_hard, 1.0 - prob_hard)
        
        # Routing Logic
        use_hard = (conf_hard >= threshold)    # Exit 1: >= 99.9%
        use_easy = (~use_hard)                 # Exit 2: Below 99.9%

        # Select the winning logit
        final_logits = torch.zeros_like(logits_hard)
        final_logits[use_hard] = logits_hard[use_hard]
        final_logits[use_easy] = logits_easy[use_easy]

        return final_logits

# ==========================================
# 3. TRAINING CONFIGURATION
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {str(device).upper()}")

    print("[INIT] Loading Face Detector...")
    detector = FaceDetector(device=device, embed=True)

    print("[INIT] Loading Cached Embeddings directly to RAM...")
    cache_path = "D:\\cctv\\watch_dogs\\datasets\\bollywood_faces\\embedding_cache(1,3).pt"
    
    if not os.path.exists(cache_path):
        print("ERROR: Cache not found. Run cache_dataset.py first.")
        sys.exit(1)

    cached_data = torch.load(cache_path)
    
    dataset = torch.utils.data.TensorDataset(
        cached_data['e1'], 
        cached_data['e2'], 
        cached_data['y']
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )
    
    model = AttentionHead(dim=512).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scaler = torch.amp.GradScaler('cuda')

    # Saved to a new specific directory
    checkpoint_dir = os.path.join(REPO_DIR, "trained", "bollywood_faces", "v2.32_with_512_margin_exit")
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_file = os.path.join(checkpoint_dir, "training_history.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s | %(message)s')
    logging.info(f"--- Started Training 2-Exit Cascade on {device} ---")

    # ==========================================
    # 4. HIGH-PERFORMANCE TRAINING LOOP
    # ==========================================
    EPOCHS = 4
    MARGIN_HARD = 0.25
    MARGIN_EASY = 0.000

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

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

        avg_loss = total_loss / len(loader)
        summary_msg = f"Epoch {epoch+1}/{EPOCHS} Completed | Avg Train Loss: {avg_loss:.4f}"
        print(f"\n=== {summary_msg} ===")
        logging.info(summary_msg)

        save_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_path)

    print("Training Complete.")
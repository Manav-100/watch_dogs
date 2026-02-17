import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.attention_head import AttentionHead
from model.dataset import PairDataset
from siamese_model import FaceDetector
import os

# --- SETUP ---
pwd = os.getcwd()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

# 1. Initialize Encoder
detector = FaceDetector(device=device, embed=True)

# 2. Dataset
dataset = PairDataset(
    parent_dir=r"d:\College\Vscode\watchdogs\datasets\bollywood_faces\cropped", 
    encoder=detector.app.models['recognition']
)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 3. Model
model = AttentionHead().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 4. Checkpoints
checkpoint_dir = os.path.join(pwd, "trained", "bollywood_faces", "v1")
os.makedirs(checkpoint_dir, exist_ok=True)

# --- TRAINING LOOP ---
EPOCHS = 4

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for i, (e1, e2, y) in enumerate(loader):
        e1, e2, y = e1.to(device), e2.to(device), y.to(device)
        
        # 1. Normalize Embeddings (CRITICAL for ArcFace)
        e1 = F.normalize(e1, p=2, dim=1)
        e2 = F.normalize(e2, p=2, dim=1)
        
        # 2. Forward Pass
        logits = model(e1, e2)
        
        # 3. Calculate Loss (FIXED SHAPE MISMATCH)
        # We flatten both to [Batch_Size] to ensure they match perfectly
        loss = criterion(logits.view(-1), y.float().view(-1))

        # 4. Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{i}/{len(loader)}] | Loss: {loss.item():.4f}", flush=True)

    # Epoch Summary
    avg_loss = total_loss / len(loader)
    print(f"=== Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f} ===")
    
    # Save Checkpoint
    save_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved checkpoint: {save_path}\n")

print("Training Complete.")
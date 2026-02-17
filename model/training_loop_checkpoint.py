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

# --- NEW CONFIGURATION ---
# Path to the model you want to load (set to None if starting fresh)
RESUME_CHECKPOINT = os.path.join(pwd, "trained", "bollywood_faces", "v1", "epoch_3.pth") 

# New lower learning rate (e.g., 1e-5 is 10x smaller than your original 1e-4)
NEW_LR = 1e-5 

# How many *additional* epochs you want to train
ADDITIONAL_EPOCHS = 16 

# 1. Initialize Encoder
detector = FaceDetector(device=device, embed=True)

# 2. Dataset
dataset = PairDataset(
    parent_dir=r"d:\College\Vscode\watchdogs\datasets\ms1m_arcface\raw", 
    encoder=detector.app.models['recognition']
)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 3. Model
model = AttentionHead().to(device)

# --- LOAD CHECKPOINT LOGIC ---
start_epoch = 0

if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
    print(f"Loading checkpoint from: {RESUME_CHECKPOINT}")
    # Load the weights into the model
    model.load_state_dict(torch.load(RESUME_CHECKPOINT, map_location=device))
    
    # Extract the epoch number from the filename so we don't overwrite files
    # Assuming format "epoch_X.pth"
    try:
        filename = os.path.basename(RESUME_CHECKPOINT)
        last_epoch_num = int(filename.split('_')[1].split('.')[0])
        start_epoch = last_epoch_num + 1
        print(f"Resuming from Epoch {start_epoch}")
    except:
        print("Could not parse epoch number. Starting count from 0.")
else:
    print("No checkpoint found or specified. Starting fresh training.")

# 4. Optimizer & Loss
criterion = nn.BCEWithLogitsLoss()

# Initialize Optimizer with the NEW LOWER LEARNING RATE
optimizer = torch.optim.Adam(model.parameters(), lr=NEW_LR)

# 5. Checkpoints Directory
checkpoint_dir = os.path.join(pwd, "trained", "bollywood_faces", "v2")
os.makedirs(checkpoint_dir, exist_ok=True)

# --- TRAINING LOOP ---
# Update range to start from the correct epoch
TOTAL_EPOCHS = start_epoch + ADDITIONAL_EPOCHS

print(f"Training for epochs {start_epoch} to {TOTAL_EPOCHS}")

for epoch in range(start_epoch, TOTAL_EPOCHS):
    model.train()
    total_loss = 0
    
    for i, (e1, e2, y) in enumerate(loader):
        e1, e2, y = e1.to(device), e2.to(device), y.to(device)
        
        # 1. Normalize Embeddings
        e1 = F.normalize(e1, p=2, dim=1)
        e2 = F.normalize(e2, p=2, dim=1)
        
        # 2. Forward Pass
        logits = model(e1, e2)
        
        # 3. Calculate Loss
        loss = criterion(logits.view(-1), y.float().view(-1))

        # 4. Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        if i % 10 == 0:
            print(f"Epoch [{epoch}/{TOTAL_EPOCHS}] Batch [{i}/{len(loader)}] | Loss: {loss.item():.4f}", flush=True)

    # Epoch Summary
    avg_loss = total_loss / len(loader)
    print(f"=== Epoch {epoch} Completed | Average Loss: {avg_loss:.4f} ===")
    
    # Save Checkpoint
    save_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved checkpoint: {save_path}\n")

print("Fine-tuning Complete.")
import random
import cv2
from torch.utils.data import Dataset
import torch
from model.distortion_simulation import custom_distortion_pipeline
import os
import numpy as np

class PairDataset(Dataset):
    def __init__(self, parent_dir, encoder):
        self.ids = []
        self.image_index = {}
        
        # Build Index
        for person_folder in os.listdir(parent_dir):
            person_path = os.path.join(parent_dir, person_folder)
            if os.path.isdir(person_path):
                files = [
                    os.path.join(person_path, f) 
                    for f in os.listdir(person_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
                ]
                # Only include people with at least 2 images for positive pairs
                if len(files) >= 2:
                    self.ids.append(person_folder)
                    self.image_index[person_folder] = files
        
        self.encoder = encoder
        self.distorter = custom_distortion_pipeline

    def __len__(self):
        return 100000  

    def _load(self, path):
        # FIX: Keep as BGR! Do NOT convert to RGB.
        # InsightFace expects BGR inputs.
        return cv2.imread(path)

    def __getitem__(self, idx):
        same = random.random() < 0.5

        if same:
            pid = random.choice(self.ids)
            path1, path2 = random.sample(self.image_index[pid], 2)
            label = 1.0
        else:
            pid1, pid2 = random.sample(self.ids, 2)
            path1 = random.choice(self.image_index[pid1])
            path2 = random.choice(self.image_index[pid2])
            label = 0.0

        img1 = self._load(path1)
        # Load img2 and apply distortion
        # Note: Distortion pipeline should be robust to BGR, or just noise/blur which is color-agnostic
        img2_raw = self._load(path2)
        
        if img1 is None or img2_raw is None:
            return self.__getitem__(idx) # Retry if load fails

        try:
            # Apply distortion
            img2 = self.distorter(image=img2_raw)["image"]
        except Exception as e:
            # Fallback if distortion fails
            img2 = img2_raw

        # Get Embeddings (Now working on correct BGR images)
        # Note: We rely on train.py to do the L2 Normalization (F.normalize)
        e1 = self.encoder.get_feat(img1)
        e2 = self.encoder.get_feat(img2)

        if e1 is None or e2 is None:
            return self.__getitem__(idx)

        return (
            torch.from_numpy(e1).float().flatten(),
            torch.from_numpy(e2).float().flatten(),
            torch.tensor(label, dtype=torch.float32)
        )
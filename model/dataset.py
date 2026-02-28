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
        # Load as BGR (OpenCV Default)
        img = cv2.imread(path)
        return img

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

        img1_bgr = self._load(path1)
        img2_raw_bgr = self._load(path2)
        
        if img1_bgr is None or img2_raw_bgr is None:
            return self.__getitem__(idx) # Retry if load fails

        try:
            # 1. Convert BGR to RGB for the distortion pipeline
            img2_rgb = cv2.cvtColor(img2_raw_bgr, cv2.COLOR_BGR2RGB)
            
            # 2. Apply distortion
            distorted_rgb = self.distorter(image=img2_rgb)["image"]
            
            # 3. Convert back to BGR for InsightFace
            img2_bgr = cv2.cvtColor(distorted_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            # Fallback if distortion fails
            img2_bgr = img2_raw_bgr

        # Get Embeddings (Strictly using BGR images)
        e1 = self.encoder.get_feat(img1_bgr)
        e2 = self.encoder.get_feat(img2_bgr)

        if e1 is None or e2 is None:
            return self.__getitem__(idx)

        return (
            torch.from_numpy(e1).float().flatten(),
            torch.from_numpy(e2).float().flatten(),
            torch.tensor(label, dtype=torch.float32)
        )
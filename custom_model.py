import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class Watch_Dogs(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        # Register head modules correctly
        for name, module in head.named_children():
            setattr(self, name, module)
        self._head_forward = head.forward 

    def forward(self, img1, img2):
        # 1. Get Embeddings
        e1 = self.encoder(img1)
        e2 = self.encoder(img2)
        
        # 2. Ensure Batch Dimension [1, 512]
        if e1.dim() == 1: e1 = e1.unsqueeze(0)
        if e2.dim() == 1: e2 = e2.unsqueeze(0)
            
        # 3. L2 Normalize (Essential for Face Recognition consistency)
        e1 = F.normalize(e1, p=2, dim=1)
        e2 = F.normalize(e2, p=2, dim=1)
        
        return self._head_forward(e1, e2)

class FaceEncoder(nn.Module):
    def __init__(self, onnx_model):
        super().__init__()
        self.model = onnx_model

    def forward(self, x):
        # Convert Tensor [1, 3, 112, 112] back to Numpy [112, 112, 3] for ONNX
        if isinstance(x, torch.Tensor):
            if x.dim() == 4: x = x.squeeze(0)
            x = x.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        # MANDATORY: Flip BGR (OpenCV) to RGB (ArcFace training distribution)
        x_rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        # get_feat handles internal 127.5 scaling
        emb = self.model.get_feat(x_rgb)
        return torch.from_numpy(emb).float().flatten()
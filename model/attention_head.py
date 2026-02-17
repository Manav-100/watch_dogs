import torch.nn as nn
import torch

class AttentionHead(nn.Module):
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
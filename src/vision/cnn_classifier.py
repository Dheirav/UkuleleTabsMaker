from typing import Optional, Tuple
import os

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    F = None


if torch is not None:
    class SmallDigitCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc1 = nn.Linear(64 * 3 * 3, 128)
            self.fc2 = nn.Linear(128, 13)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)
else:
    SmallDigitCNN = None


def load_cnn_classifier(weights_path: str):
    if torch is None:
        return None
    if not os.path.exists(weights_path):
        return None
    model = SmallDigitCNN()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict_digit(model, patch: np.ndarray) -> Optional[Tuple[int, float]]:
    if model is None or torch is None:
        return None
    if len(patch.shape) == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    patch = cv2.resize(patch, (28, 28))
    patch = patch.astype(np.float32)
    if patch.mean() > 127.5:
        patch = 255.0 - patch
    patch = patch / 255.0
    tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
    return int(idx.item()), float(conf.item())

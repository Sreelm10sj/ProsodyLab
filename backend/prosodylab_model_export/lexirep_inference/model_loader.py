"""
model_loader.py
Safe checkpoint loader for collaborator PyTorch LexiRep models and prototype vectors.
"""

import os
import json
from typing import Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn


class LexiRepEncoder(nn.Module):
    """
    Collaborator 5-layer representation learning encoder:
    [768 -> 128 -> 64 -> 32 -> 16 -> 10] with ReLU and Sigmoid bottleneck.
    """
    def __init__(self, in_dim: int = 768, hidden: Tuple[int, ...] = (128, 64, 32, 16, 10)):
        super().__init__()
        layers = []
        cur = in_dim
        for h in hidden[:-1]:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.ReLU())
            cur = h
        layers.append(nn.Linear(cur, hidden[-1]))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModelLoadError(RuntimeError):
    """Raised when model weights or prototype files cannot be loaded."""
    pass


def load_lexirep_bundle(
    language: str,
    bundle_root: Optional[str] = None,
    device: Optional[str] = None
) -> Tuple[LexiRepEncoder, np.ndarray, np.ndarray, Dict]:
    """
    Loads the trained encoder model and prototype vectors for the specified language.

    Returns:
        encoder: PyTorch LexiRepEncoder in eval mode on target device.
        z_s: (1, 10) float32 array representing the stressed prototype.
        z_u: (1, 10) float32 array representing the unstressed prototype.
        config: Dict containing full model metadata.
    """
    lang = language.upper()
    if lang not in ["GER", "ITA"]:
        raise ValueError(f"Unsupported language '{language}'. Supported languages: 'GER', 'ITA'.")

    if bundle_root is None:
        bundle_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), lang)

    if not os.path.isdir(bundle_root):
        raise ModelLoadError(f"Model bundle directory not found: {bundle_root}")

    # File paths
    config_path = os.path.join(bundle_root, "config.json")
    model_path = os.path.join(bundle_root, "model", "final_lexirep_model.pt")
    proto_path = os.path.join(bundle_root, "anchors", "prototypes_10d.npy")

    if not os.path.isfile(config_path):
        raise ModelLoadError(f"Configuration file missing: {config_path}")
    if not os.path.isfile(model_path):
        raise ModelLoadError(f"Model checkpoint file missing: {model_path}")
    if not os.path.isfile(proto_path):
        raise ModelLoadError(f"Prototype vectors missing: {proto_path}")

    # Load configuration
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Resolve device
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    # Instantiate encoder
    encoder = LexiRepEncoder(
        in_dim=config.get("input_dim", 768),
        hidden=tuple(config.get("layer_units", [128, 64, 32, 16, 10]))
    )

    # Load weights safely
    state_dict = torch.load(model_path, map_location=target_device, weights_only=True)
    encoder.load_state_dict(state_dict)
    encoder.to(target_device)
    encoder.eval()

    # Load prototypes
    proto_arr = np.load(proto_path)  # (2, 10)
    z_s = proto_arr[0:1].astype(np.float32)
    z_u = proto_arr[1:2].astype(np.float32)

    return encoder, z_s, z_u, config

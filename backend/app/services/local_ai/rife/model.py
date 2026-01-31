"""
RIFE Model wrapper for inference
Real-Time Intermediate Flow Estimation for Video Frame Interpolation
"""

import torch
import torch.nn as nn
from pathlib import Path
from .IFNet import IFNet


class Model:
    """
    RIFE Model wrapper for frame interpolation inference.

    This class provides a simple interface to load pre-trained RIFE models
    and perform frame interpolation between two images.
    """

    def __init__(self, device=None):
        """
        Initialize RIFE model.

        Args:
            device: torch device (cuda/cpu). If None, auto-detect.
        """
        self.flownet = IFNet()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, path: str, rank: int = 0):
        """
        Load pre-trained model weights.

        Args:
            path: Path to model weights file (.pth) or directory containing flownet.pkl
            rank: GPU rank (for distributed training compatibility)
        """
        def convert(param):
            """Convert DDP model keys to regular model keys"""
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k or "module." not in k
            }

        path = Path(path)

        if path.is_dir():
            # Legacy format: directory with flownet.pkl
            model_path = path / "flownet.pkl"
        elif path.suffix == '.pth':
            model_path = path
        else:
            model_path = path

        if rank <= 0:
            state_dict = torch.load(str(model_path), map_location=self.device)
            # Handle both DDP and non-DDP checkpoints
            converted = convert(state_dict)
            self.flownet.load_state_dict(converted, strict=False)

    def eval(self):
        """Set model to evaluation mode"""
        self.flownet.eval()

    def train(self):
        """Set model to training mode"""
        self.flownet.train()

    def to(self, device):
        """Move model to device"""
        self.device = device
        self.flownet.to(device)
        return self

    def inference(self, img0, img1, scale=1, scale_list=None, TTA=False, timestep=0.5):
        """
        Perform frame interpolation between two images.

        Args:
            img0: First frame tensor [B, 3, H, W], values in [0, 1]
            img1: Second frame tensor [B, 3, H, W], values in [0, 1]
            scale: Overall scale factor
            scale_list: Multi-scale factors [default: [4, 2, 1]]
            TTA: Test-time augmentation (flip + average)
            timestep: Interpolation position (0.0 = img0, 1.0 = img1, 0.5 = middle)

        Returns:
            Interpolated frame tensor [B, 3, H, W]
        """
        if scale_list is None:
            scale_list = [4, 2, 1]

        # Adjust scale list
        adjusted_scale_list = [s * 1.0 / scale for s in scale_list]

        # Concatenate input images (no GT during inference)
        imgs = torch.cat((img0, img1), 1)

        with torch.no_grad():
            # Add empty channel for GT placeholder
            x = torch.cat((imgs, torch.zeros_like(img0[:, :0])), 1)

            flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(
                x, adjusted_scale_list, timestep=timestep
            )

            if not TTA:
                return merged[2]
            else:
                # Test-time augmentation with flip
                x_flip = torch.cat((imgs.flip(2).flip(3), torch.zeros_like(img0[:, :0])), 1)
                flow2, mask2, merged2, _, _, _ = self.flownet(
                    x_flip, adjusted_scale_list, timestep=timestep
                )
                return (merged[2] + merged2[2].flip(2).flip(3)) / 2

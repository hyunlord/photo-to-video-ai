#!/usr/bin/env python3
"""
Model Download Script for Local AI

Downloads Stable Video Diffusion models from HuggingFace Hub.
Supports caching and resume for large model files.

Usage:
    python scripts/download_models.py --model svd-xt-1.1
    python scripts/download_models.py --model svd --output ./models
    python scripts/download_models.py --list
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    print("Error: huggingface_hub package not installed.")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)


# Model configurations
MODELS = {
    "svd": {
        "repo_id": "stabilityai/stable-video-diffusion-img2vid",
        "description": "SVD base model - 14 frames (~2 seconds)",
        "size": "~9GB",
        "vram_required": "6GB+",
    },
    "svd-xt": {
        "repo_id": "stabilityai/stable-video-diffusion-img2vid-xt",
        "description": "SVD-XT extended model - 25 frames (~4 seconds)",
        "size": "~9GB",
        "vram_required": "8GB+",
    },
    "svd-xt-1.1": {
        "repo_id": "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        "description": "SVD-XT 1.1 improved model - 25 frames (~4 seconds) [RECOMMENDED]",
        "size": "~9.5GB",
        "vram_required": "8GB+",
    },
}

# Default download location
DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "backend" / "models"


def list_models():
    """Print available models."""
    print("\nAvailable Models:")
    print("-" * 70)
    for name, info in MODELS.items():
        print(f"\n  {name}:")
        print(f"    Repository: {info['repo_id']}")
        print(f"    Description: {info['description']}")
        print(f"    Size: {info['size']}")
        print(f"    VRAM Required: {info['vram_required']}")
    print("\n" + "-" * 70)
    print("\nRecommended: svd-xt-1.1 (best quality)")


def check_hf_token() -> Optional[str]:
    """Check for HuggingFace token."""
    # Check environment variable
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token

    # Check if logged in via CLI
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            return token
    except Exception:
        pass

    return None


def download_model(
    model_name: str,
    output_dir: Path,
    token: Optional[str] = None,
    resume: bool = True
) -> Path:
    """
    Download a model from HuggingFace Hub.

    Args:
        model_name: Name of the model (e.g., 'svd-xt-1.1')
        output_dir: Directory to save the model
        token: HuggingFace token (optional, for private repos)
        resume: Resume incomplete downloads

    Returns:
        Path to the downloaded model directory
    """
    if model_name not in MODELS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {', '.join(MODELS.keys())}")
        sys.exit(1)

    model_info = MODELS[model_name]
    repo_id = model_info["repo_id"]

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = output_dir / model_name

    print(f"\nDownloading: {model_name}")
    print(f"Repository: {repo_id}")
    print(f"Size: {model_info['size']}")
    print(f"Destination: {model_dir}")
    print("-" * 50)

    try:
        # Download the entire model repository
        local_dir = snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            token=token,
            resume_download=resume,
            # Only download essential files (skip large unnecessary files)
            ignore_patterns=[
                "*.md",
                "*.txt",
                ".gitattributes",
            ],
        )

        print(f"\nDownload complete!")
        print(f"Model saved to: {local_dir}")

        # Verify essential files exist
        essential_files = [
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/diffusion_pytorch_model.fp16.safetensors",
        ]

        missing_files = []
        for f in essential_files:
            if not (model_dir / f).exists():
                # Check for alternative file names
                alt_f = f.replace(".fp16", "")
                if not (model_dir / alt_f).exists():
                    missing_files.append(f)

        if missing_files:
            print(f"\nWarning: Some expected files are missing:")
            for f in missing_files:
                print(f"  - {f}")
            print("The model may still work if alternative formats exist.")

        return Path(local_dir)

    except RepositoryNotFoundError:
        print(f"\nError: Repository not found: {repo_id}")
        print("This may be because:")
        print("  1. The repository doesn't exist")
        print("  2. You need to accept the model's license on HuggingFace")
        print("  3. You need to provide an authentication token")
        print(f"\nVisit: https://huggingface.co/{repo_id}")
        sys.exit(1)

    except Exception as e:
        print(f"\nError downloading model: {e}")
        if "401" in str(e) or "403" in str(e):
            print("\nAuthentication required. Please:")
            print("  1. Create an account at https://huggingface.co")
            print("  2. Accept the model license")
            print("  3. Create an access token")
            print("  4. Run: huggingface-cli login")
            print("     Or set HF_TOKEN environment variable")
        sys.exit(1)


def verify_model(model_dir: Path) -> bool:
    """Verify that a downloaded model is valid."""
    if not model_dir.exists():
        return False

    # Check for essential directories
    required_dirs = ["unet", "vae", "scheduler", "image_encoder", "feature_extractor"]
    for d in required_dirs:
        if not (model_dir / d).exists():
            print(f"Warning: Missing directory: {d}")
            return False

    # Check for model weights
    unet_weights = list((model_dir / "unet").glob("*.safetensors"))
    if not unet_weights:
        unet_weights = list((model_dir / "unet").glob("*.bin"))

    if not unet_weights:
        print("Warning: No UNet weights found")
        return False

    print(f"Model verification passed: {model_dir.name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download Stable Video Diffusion models for local AI processing"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=list(MODELS.keys()),
        help="Model to download"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help=f"Output directory (default: {DEFAULT_MODEL_DIR})"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing model download"
    )
    parser.add_argument(
        "--token", "-t",
        type=str,
        help="HuggingFace token (or set HF_TOKEN env var)"
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if not args.model:
        parser.print_help()
        print("\nExample: python scripts/download_models.py --model svd-xt-1.1")
        return

    output_dir = Path(args.output)
    model_dir = output_dir / args.model

    if args.verify:
        if verify_model(model_dir):
            print(f"Model '{args.model}' is valid and ready to use.")
        else:
            print(f"Model '{args.model}' verification failed or not found.")
        return

    # Get token
    token = args.token or check_hf_token()
    if not token:
        print("Note: No HuggingFace token found.")
        print("Some models may require authentication.")
        print("Run 'huggingface-cli login' if download fails.")

    # Download the model
    download_model(args.model, output_dir, token)

    # Verify after download
    print("\nVerifying download...")
    if verify_model(model_dir):
        print("\nModel is ready to use!")
        print(f"\nTo use this model, set in your .env:")
        print(f"  LOCAL_MODEL_PATH={model_dir}")
    else:
        print("\nWarning: Verification found issues. Model may not work correctly.")


if __name__ == "__main__":
    main()

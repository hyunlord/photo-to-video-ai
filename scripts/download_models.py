#!/usr/bin/env python3
"""
AI 모델 다운로드 스크립트

Wan 2.1 FLF2V 및 관련 모델 다운로드

Usage:
    python scripts/download_models.py              # FP16 버전 (기본)
    python scripts/download_models.py --fp8        # FP8 양자화 버전 (저용량)
    python scripts/download_models.py --vision     # Vision 모델 포함
    python scripts/download_models.py --all        # 모든 모델 (후처리 포함)
"""

import os
import sys
import argparse
from pathlib import Path


def check_dependencies():
    """필요한 패키지 확인"""
    try:
        from huggingface_hub import snapshot_download
        return True
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        return True


def download_wan_model(use_fp8: bool = False, models_dir: Path = None):
    """
    Wan 2.1 FLF2V 모델 다운로드

    Args:
        use_fp8: True면 FP8 양자화 버전 (~16GB), False면 FP16 버전 (~33GB)
        models_dir: 모델 저장 디렉토리
    """
    from huggingface_hub import snapshot_download

    if models_dir is None:
        models_dir = Path("backend/models")

    model_dir = models_dir / "wan" / "wan2.1_flf2v_720p"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_id = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"

    print("=" * 60)
    print("Wan 2.1 FLF2V Model Download")
    print("=" * 60)
    print(f"Model ID: {model_id}")
    print(f"Destination: {model_dir}")

    if use_fp8:
        print("Version: FP8 Quantized (~16GB)")
        print("Note: Recommended for GPUs with <12GB VRAM")
        revision = "fp8"
    else:
        print("Version: FP16 Full (~33GB)")
        print("Note: Recommended for GPUs with 16GB+ VRAM")
        revision = "main"

    print("-" * 60)
    print("Downloading... (this may take a while)")

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(model_dir),
            revision=revision,
            resume_download=True,
        )
        print(f"\n✓ Wan 2.1 FLF2V downloaded successfully!")
        print(f"  Location: {model_dir}")
        return True

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False


def download_vision_model(models_dir: Path = None):
    """
    BLIP Vision 모델 다운로드 (프롬프트 자동 생성용)
    """
    from huggingface_hub import snapshot_download

    if models_dir is None:
        models_dir = Path("backend/models")

    model_dir = models_dir / "blip"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_id = "Salesforce/blip-image-captioning-base"

    print("=" * 60)
    print("BLIP Vision Model Download")
    print("=" * 60)
    print(f"Model ID: {model_id}")
    print(f"Destination: {model_dir}")
    print("-" * 60)

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(model_dir),
            resume_download=True,
        )
        print(f"\n✓ BLIP model downloaded successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False


def download_postprocess_models(models_dir: Path = None):
    """
    후처리 모델 다운로드 (Real-ESRGAN, GFPGAN)

    Note: 후처리는 현재 선택적 기능
    """
    from huggingface_hub import hf_hub_download

    if models_dir is None:
        models_dir = Path("backend/models")

    postprocess_dir = models_dir / "postprocess"
    postprocess_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Post-processing Models Download")
    print("=" * 60)

    # Real-ESRGAN
    print("\nDownloading Real-ESRGAN...")
    try:
        hf_hub_download(
            repo_id="ai-forever/Real-ESRGAN",
            filename="RealESRGAN_x4plus.pth",
            local_dir=str(postprocess_dir / "realesrgan"),
        )
        print("✓ Real-ESRGAN downloaded")
    except Exception as e:
        print(f"✗ Real-ESRGAN download failed: {e}")

    # GFPGAN (얼굴 향상)
    print("\nDownloading GFPGAN...")
    try:
        hf_hub_download(
            repo_id="TencentARC/GFPGAN",
            filename="GFPGANv1.4.pth",
            local_dir=str(postprocess_dir / "gfpgan"),
        )
        print("✓ GFPGAN downloaded")
    except Exception as e:
        print(f"✗ GFPGAN download failed: {e}")

    print(f"\n✓ Post-processing models saved to: {postprocess_dir}")
    return True


def verify_installation(models_dir: Path = None):
    """설치 확인"""
    if models_dir is None:
        models_dir = Path("backend/models")

    print("\n" + "=" * 60)
    print("Installation Verification")
    print("=" * 60)

    # Wan 모델 확인
    wan_dir = models_dir / "wan" / "wan2.1_flf2v_720p"
    wan_exists = wan_dir.exists() and any(wan_dir.iterdir()) if wan_dir.exists() else False
    print(f"Wan 2.1 FLF2V: {'✓ Installed' if wan_exists else '✗ Not found'}")

    # BLIP 모델 확인
    blip_dir = models_dir / "blip"
    blip_exists = blip_dir.exists() and any(blip_dir.iterdir()) if blip_dir.exists() else False
    print(f"BLIP Vision:   {'✓ Installed' if blip_exists else '○ Optional (not installed)'}")

    # 후처리 모델 확인
    postprocess_dir = models_dir / "postprocess"
    postprocess_exists = postprocess_dir.exists() and any(postprocess_dir.iterdir()) if postprocess_dir.exists() else False
    print(f"Post-process:  {'✓ Installed' if postprocess_exists else '○ Optional (not installed)'}")

    # GPU 확인
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\nGPU: {gpu_name} ({vram_gb:.1f}GB VRAM)")

            if vram_gb < 8:
                print("⚠ Warning: Low VRAM. Consider using FP8 version.")
            elif vram_gb < 12:
                print("ℹ Recommended: Use FP8 version for best performance.")
            else:
                print("✓ VRAM sufficient for FP16 version.")
        else:
            print("\n⚠ Warning: CUDA not available. GPU required for video generation.")
    except ImportError:
        print("\n⚠ Warning: PyTorch not installed. Cannot verify GPU.")

    return wan_exists


def main():
    parser = argparse.ArgumentParser(
        description="Download AI models for Photo-to-Video generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_models.py              # Download FP16 Wan model
    python scripts/download_models.py --fp8        # Download FP8 Wan model (smaller)
    python scripts/download_models.py --vision     # Also download BLIP for auto-prompts
    python scripts/download_models.py --all        # Download all models

Model Sizes:
    Wan 2.1 FLF2V (FP16): ~33GB
    Wan 2.1 FLF2V (FP8):  ~16GB
    BLIP Vision:          ~1GB
    Post-processing:      ~500MB
        """
    )

    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Download FP8 quantized version (smaller, for <12GB VRAM)"
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Also download BLIP vision model for auto-prompt generation"
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Also download post-processing models (Real-ESRGAN, GFPGAN)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all models (Wan + Vision + Post-process)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify installation, don't download"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="backend/models",
        help="Models directory (default: backend/models)"
    )

    args = parser.parse_args()
    models_dir = Path(args.models_dir)

    # 검증만 수행
    if args.verify_only:
        verify_installation(models_dir)
        return

    # 의존성 확인
    check_dependencies()

    success = True

    # Wan 모델 다운로드 (필수)
    if not download_wan_model(use_fp8=args.fp8, models_dir=models_dir):
        success = False

    # Vision 모델 다운로드 (선택)
    if args.vision or args.all:
        download_vision_model(models_dir)

    # 후처리 모델 다운로드 (선택)
    if args.postprocess or args.all:
        download_postprocess_models(models_dir)

    # 설치 확인
    verify_installation(models_dir)

    if success:
        print("\n" + "=" * 60)
        print("Setup Complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Start the backend: cd docker && docker-compose up")
        print("2. Upload photos in the web interface")
        print("3. Generate your first video!")
    else:
        print("\n⚠ Some downloads failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

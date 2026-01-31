"""
AI Prompt Enhancer
사용자 프롬프트를 AI가 자동으로 보강

Supports three modes:
- none: Use user prompt as-is
- append: Add quality/motion enhancements to user prompt
- auto: AI analyzes images and generates optimal prompt
"""

import torch
from typing import List, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class PromptEnhancer:
    """
    이미지 분석 기반 프롬프트 자동 보강

    옵션:
    1. none - 사용자 프롬프트만 사용
    2. append - 사용자 프롬프트 + 기본 보강
    3. auto - AI가 이미지 분석 후 최적 프롬프트 생성
    """

    # 기본 보강 키워드
    MOTION_ENHANCEMENTS = [
        "smooth natural motion",
        "fluid movement",
        "cinematic transition",
    ]

    QUALITY_ENHANCEMENTS = [
        "high quality",
        "professional cinematography",
        "natural lighting",
    ]

    STYLE_ENHANCEMENTS = [
        "photorealistic",
        "sharp focus",
        "natural colors",
    ]

    def __init__(self, use_vision_model: bool = False):
        """
        Args:
            use_vision_model: True면 이미지 분석 모델 사용 (추가 VRAM 필요)
        """
        self.use_vision_model = use_vision_model
        self.vision_model = None
        self.vision_processor = None
        self._vision_initialized = False

    def _get_device(self) -> str:
        """Get best available device string (CUDA > MPS > CPU)"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _clear_memory_cache(self):
        """Clear GPU/MPS memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

    def _load_vision_model(self):
        """Vision 모델 lazy loading (CUDA/MPS/CPU 지원)"""
        if self._vision_initialized:
            return

        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration

            logger.info("Loading BLIP vision model...")

            self.vision_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.vision_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )

            device = self._get_device()
            self.vision_model = self.vision_model.to(device)
            self._vision_initialized = True

            logger.info(f"BLIP vision model loaded on {device}")

        except ImportError:
            logger.warning("transformers not installed, vision model unavailable")
            self.use_vision_model = False
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
            self.use_vision_model = False

    def enhance_prompt(
        self,
        user_prompt: str,
        first_image: Optional[Image.Image] = None,
        last_image: Optional[Image.Image] = None,
        enhancement_mode: str = "append"
    ) -> str:
        """
        프롬프트 보강

        Args:
            user_prompt: 사용자 입력 프롬프트
            first_image: 시작 이미지 (이미지 분석용)
            last_image: 끝 이미지 (이미지 분석용)
            enhancement_mode:
                - "none": 사용자 프롬프트 그대로
                - "append": 사용자 프롬프트 + 기본 보강
                - "auto": AI가 이미지 분석 후 최적 프롬프트 생성

        Returns:
            보강된 프롬프트
        """
        if enhancement_mode == "none":
            return user_prompt or "smooth transition between two images"

        if enhancement_mode == "append":
            return self._append_enhancements(user_prompt)

        if enhancement_mode == "auto":
            return self._auto_enhance(user_prompt, first_image, last_image)

        # Default fallback
        return self._append_enhancements(user_prompt)

    def _append_enhancements(self, user_prompt: str) -> str:
        """기본 보강 추가"""
        base = user_prompt.strip() if user_prompt else "smooth transition"

        enhancements = [
            self.MOTION_ENHANCEMENTS[0],  # smooth natural motion
            self.QUALITY_ENHANCEMENTS[0],  # high quality
            "cinematic",
        ]

        return f"{base}, {', '.join(enhancements)}"

    def _auto_enhance(
        self,
        user_prompt: str,
        first_image: Optional[Image.Image],
        last_image: Optional[Image.Image]
    ) -> str:
        """이미지 분석 기반 자동 보강"""

        # Vision 모델 없거나 이미지 없으면 기본 보강
        if not self.use_vision_model or first_image is None:
            return self._append_enhancements(user_prompt)

        # Vision 모델로 이미지 분석
        try:
            description = self._analyze_image(first_image)

            if user_prompt:
                return f"{user_prompt}, {description}, smooth natural motion, high quality"
            else:
                return f"smooth transition showing {description}, cinematic, high quality"

        except Exception as e:
            logger.warning(f"Image analysis failed: {e}")
            return self._append_enhancements(user_prompt)

    def _analyze_image(self, image: Image.Image) -> str:
        """이미지 분석 (Vision 모델 사용)"""
        self._load_vision_model()

        if not self._vision_initialized:
            return "natural scene"

        # 이미지 전처리
        inputs = self.vision_processor(image, return_tensors="pt")
        inputs = {k: v.to(self.vision_model.device) for k, v in inputs.items()}

        # 캡션 생성
        with torch.no_grad():
            output = self.vision_model.generate(
                **inputs,
                max_length=50,
                num_beams=4,
            )

        caption = self.vision_processor.decode(output[0], skip_special_tokens=True)

        return f"{caption}, natural movement"

    def generate_prompts_for_sequence(
        self,
        images: List[Image.Image],
        user_prompt: str = "",
        enhancement_mode: str = "append"
    ) -> List[str]:
        """
        이미지 시퀀스에 대한 프롬프트 리스트 생성

        Args:
            images: 이미지 리스트
            user_prompt: 전체에 적용할 사용자 프롬프트
            enhancement_mode: 보강 모드 ("none", "append", "auto")

        Returns:
            각 전환에 대한 프롬프트 리스트 (len = len(images) - 1)
        """
        if len(images) < 2:
            return []

        prompts = []

        for i in range(len(images) - 1):
            prompt = self.enhance_prompt(
                user_prompt=user_prompt,
                first_image=images[i],
                last_image=images[i + 1],
                enhancement_mode=enhancement_mode
            )
            prompts.append(prompt)

        logger.info(f"Generated {len(prompts)} prompts with mode '{enhancement_mode}'")
        return prompts

    def unload_vision_model(self):
        """Vision 모델 메모리 해제 (CUDA/MPS 지원)"""
        if self.vision_model is not None:
            del self.vision_model
            del self.vision_processor
            self.vision_model = None
            self.vision_processor = None
            self._vision_initialized = False

            self._clear_memory_cache()

            logger.info("Vision model unloaded")


# Singleton
_enhancer: Optional[PromptEnhancer] = None


def get_prompt_enhancer(use_vision: bool = False) -> PromptEnhancer:
    """Get singleton PromptEnhancer instance"""
    global _enhancer
    if _enhancer is None:
        _enhancer = PromptEnhancer(use_vision_model=use_vision)
    return _enhancer

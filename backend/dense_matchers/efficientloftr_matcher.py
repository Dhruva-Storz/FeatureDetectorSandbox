"""EfficientLoFTR matcher via HuggingFace Transformers."""
import numpy as np
from typing import Optional
import logging

from .base import BaseDenseMatcher, DenseMatchResult

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_model = None
_processor = None
_torch = None
_Image = None


def _ensure_loaded():
    """Lazy load the model and dependencies."""
    global _model, _processor, _torch, _Image

    if _model is not None:
        return

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForKeypointMatching

    _torch = torch
    _Image = Image

    logger.info("Loading EfficientLoFTR model from HuggingFace...")
    _processor = AutoImageProcessor.from_pretrained("zju-community/efficientloftr")
    _model = AutoModelForKeypointMatching.from_pretrained("zju-community/efficientloftr")

    # Move to GPU if available
    if torch.cuda.is_available():
        _model = _model.cuda()
        logger.info("EfficientLoFTR model loaded on GPU")
    else:
        logger.info("EfficientLoFTR model loaded on CPU")

    _model.eval()


class EfficientLoFTRMatcher(BaseDenseMatcher):
    """EfficientLoFTR matcher using HuggingFace Transformers.

    EfficientLoFTR is an efficient detector-free local feature matching method
    that produces semi-dense matches with sparse-like speed. It's ~2.5x faster
    than the original LoFTR while achieving higher accuracy.

    Reference: https://huggingface.co/zju-community/efficientloftr
    """

    name = "EfficientLoFTR"
    description = "EfficientLoFTR - Semi-dense detector-free matcher (HuggingFace)"

    def __init__(self, default_threshold: float = 0.2):
        """Initialize the EfficientLoFTR matcher.

        Args:
            default_threshold: Default confidence threshold for filtering matches
        """
        self.default_threshold = default_threshold
        self._initialized = False

    def _initialize(self):
        """Initialize model on first use."""
        if not self._initialized:
            _ensure_loaded()
            self._initialized = True

    def match(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        threshold: float = 0.2,
    ) -> DenseMatchResult:
        """
        Match features between two images using EfficientLoFTR.

        Args:
            image1: First image (BGR or grayscale, HxWxC or HxW)
            image2: Second image (BGR or grayscale, HxWxC or HxW)
            threshold: Minimum confidence score for matches (0-1)

        Returns:
            DenseMatchResult containing matched keypoint pairs and scores
        """
        self._initialize()

        # Convert numpy arrays to PIL Images
        # EfficientLoFTR can handle grayscale or RGB
        if len(image1.shape) == 2:
            pil_image1 = _Image.fromarray(image1, mode='L').convert('RGB')
        else:
            # OpenCV uses BGR, convert to RGB
            pil_image1 = _Image.fromarray(image1[:, :, ::-1])

        if len(image2.shape) == 2:
            pil_image2 = _Image.fromarray(image2, mode='L').convert('RGB')
        else:
            pil_image2 = _Image.fromarray(image2[:, :, ::-1])

        images = [pil_image1, pil_image2]

        # Process images
        inputs = _processor(images, return_tensors="pt")

        # Move to GPU if model is on GPU
        if next(_model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Run inference
        with _torch.inference_mode():
            outputs = _model(**inputs)

        # Post-process to get keypoints and matches
        image_sizes = [[(pil_image1.height, pil_image1.width),
                        (pil_image2.height, pil_image2.width)]]

        processed = _processor.post_process_keypoint_matching(
            outputs, image_sizes, threshold=threshold
        )

        # Extract results
        result = processed[0]

        # Convert to numpy arrays
        keypoints0 = result["keypoints0"].cpu().numpy()  # Nx2
        keypoints1 = result["keypoints1"].cpu().numpy()  # Nx2
        scores = result["matching_scores"].cpu().numpy()  # N

        return DenseMatchResult(
            keypoints1=keypoints0,
            keypoints2=keypoints1,
            scores=scores,
        )

    def get_config(self) -> dict:
        return {
            "model": "zju-community/efficientloftr",
            "default_threshold": self.default_threshold,
            "gpu_available": self.is_gpu_available(),
        }

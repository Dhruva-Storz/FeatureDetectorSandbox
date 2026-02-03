"""LightGlue matchers with various feature extractors via HuggingFace Transformers."""
import numpy as np
from typing import Optional, Dict, Any
import logging

from .base import BaseDenseMatcher, DenseMatchResult

logger = logging.getLogger(__name__)

# Lazy loaded models - keyed by model name
_models: Dict[str, Any] = {}
_processors: Dict[str, Any] = {}
_torch = None
_Image = None


def _ensure_loaded(model_name: str):
    """Lazy load a specific model and dependencies."""
    global _models, _processors, _torch, _Image

    if model_name in _models:
        return

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    _torch = torch
    _Image = Image

    logger.info(f"Loading {model_name} from HuggingFace...")
    _processors[model_name] = AutoImageProcessor.from_pretrained(model_name)
    _models[model_name] = AutoModel.from_pretrained(model_name)

    # Move to GPU if available
    if torch.cuda.is_available():
        _models[model_name] = _models[model_name].cuda()
        logger.info(f"{model_name} loaded on GPU")
    else:
        logger.info(f"{model_name} loaded on CPU")

    _models[model_name].eval()


class LightGlueBaseMatcher(BaseDenseMatcher):
    """Base class for LightGlue matchers with different feature extractors.

    LightGlue is a deep neural network that learns to match local features
    across images. It can be paired with different feature extractors like
    SuperPoint, DISK, or SIFT.
    """

    # Subclasses should override these
    model_name = ""
    name = "LightGlue"
    description = "LightGlue matcher"

    def __init__(self, default_threshold: float = 0.0):
        """Initialize the LightGlue matcher.

        Args:
            default_threshold: Default confidence threshold for filtering matches
        """
        self.default_threshold = default_threshold
        self._initialized = False

    def _initialize(self):
        """Initialize model on first use."""
        if not self._initialized:
            _ensure_loaded(self.model_name)
            self._initialized = True

    def match(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        threshold: float = 0.0,
    ) -> DenseMatchResult:
        """
        Match features between two images using LightGlue.

        Args:
            image1: First image (BGR or grayscale, HxWxC or HxW)
            image2: Second image (BGR or grayscale, HxWxC or HxW)
            threshold: Minimum confidence score for matches (0-1)

        Returns:
            DenseMatchResult containing matched keypoint pairs and scores
        """
        self._initialize()

        model = _models[self.model_name]
        processor = _processors[self.model_name]

        # Convert numpy arrays to PIL Images
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
        inputs = processor(images, return_tensors="pt")

        # Move to GPU if model is on GPU
        if next(model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Run inference
        with _torch.inference_mode():
            outputs = model(**inputs)

        # Post-process to get keypoints and matches
        image_sizes = [[(pil_image1.height, pil_image1.width),
                        (pil_image2.height, pil_image2.width)]]

        processed = processor.post_process_keypoint_matching(
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
            "model": self.model_name,
            "default_threshold": self.default_threshold,
            "gpu_available": self.is_gpu_available(),
        }


class SuperPointLightGlueMatcher(LightGlueBaseMatcher):
    """SuperPoint + LightGlue matcher.

    SuperPoint is a self-supervised interest point detector and descriptor.
    Combined with LightGlue, it provides fast and accurate feature matching.

    Reference: https://huggingface.co/ETH-CVG/lightglue_superpoint
    """

    model_name = "ETH-CVG/lightglue_superpoint"
    name = "SuperPoint+LightGlue"
    description = "SuperPoint + LightGlue - Fast learned feature matcher"


class DISKLightGlueMatcher(LightGlueBaseMatcher):
    """DISK + LightGlue matcher.

    DISK (Discrete Keypoint) is a deep learning feature detector that learns
    keypoint detection and description jointly. Combined with LightGlue for matching.

    Reference: https://huggingface.co/ETH-CVG/lightglue_disk
    """

    model_name = "ETH-CVG/lightglue_disk"
    name = "DISK+LightGlue"
    description = "DISK + LightGlue - Discrete keypoint feature matcher"


# Keep backward compatibility alias
LightGlueMatcher = SuperPointLightGlueMatcher

"""Base classes for dense/end-to-end matchers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple


@dataclass
class DenseMatchResult:
    """Result of dense/end-to-end matching.

    Unlike traditional matching, dense matchers return keypoint coordinates
    directly rather than indices into a pre-computed keypoint list.
    """
    # Matched keypoint coordinates in image 1 (N x 2, x/y format)
    keypoints1: np.ndarray
    # Matched keypoint coordinates in image 2 (N x 2, x/y format)
    keypoints2: np.ndarray
    # Match confidence scores (N,)
    scores: np.ndarray
    # Optional: all detected keypoints in image 1 (before matching)
    all_keypoints1: Optional[np.ndarray] = None
    # Optional: all detected keypoints in image 2 (before matching)
    all_keypoints2: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "num_matches": len(self.scores),
            "keypoints1": self.keypoints1.tolist() if self.keypoints1 is not None else [],
            "keypoints2": self.keypoints2.tolist() if self.keypoints2 is not None else [],
            "scores": self.scores.tolist() if self.scores is not None else [],
        }

    def filter_by_score(self, threshold: float) -> "DenseMatchResult":
        """Return a new result with only matches above the score threshold."""
        mask = self.scores >= threshold
        return DenseMatchResult(
            keypoints1=self.keypoints1[mask],
            keypoints2=self.keypoints2[mask],
            scores=self.scores[mask],
            all_keypoints1=self.all_keypoints1,
            all_keypoints2=self.all_keypoints2,
        )

    def filter_by_rank(self, start: int, end: int) -> "DenseMatchResult":
        """Return matches ranked by score, from start to end index."""
        # Sort by score descending
        sorted_indices = np.argsort(-self.scores)
        selected = sorted_indices[start:end]
        return DenseMatchResult(
            keypoints1=self.keypoints1[selected],
            keypoints2=self.keypoints2[selected],
            scores=self.scores[selected],
            all_keypoints1=self.all_keypoints1,
            all_keypoints2=self.all_keypoints2,
        )


class BaseDenseMatcher(ABC):
    """Base class for dense/end-to-end matchers.

    Dense matchers take two images and directly output matched keypoint pairs,
    bypassing the traditional detect-then-match pipeline. Examples include
    LoFTR, EfficientLoFTR, and SuperPoint+LightGlue.
    """

    name: str = "BaseDenseMatcher"
    description: str = "Base dense matcher class"

    @abstractmethod
    def match(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        threshold: float = 0.0,
    ) -> DenseMatchResult:
        """
        Match features between two images end-to-end.

        Args:
            image1: First image (BGR or grayscale)
            image2: Second image (BGR or grayscale)
            threshold: Minimum confidence score for matches (0-1)

        Returns:
            DenseMatchResult containing matched keypoint pairs and scores
        """
        pass

    def get_config(self) -> dict:
        """Return current configuration."""
        return {}

    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

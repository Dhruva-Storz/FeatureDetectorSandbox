"""Base classes for feature matchers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class Match:
    """Represents a feature match."""
    query_idx: int  # Index in first image keypoints
    train_idx: int  # Index in second image keypoints
    distance: float
    img_idx: int = 0  # For multi-image matching
    
    def to_dict(self) -> dict:
        return {
            "query_idx": self.query_idx,
            "train_idx": self.train_idx,
            "distance": self.distance,
            "img_idx": self.img_idx
        }


@dataclass
class MatchResult:
    """Result of feature matching."""
    matches: list[Match]
    inlier_mask: Optional[np.ndarray] = None
    homography: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict:
        result = {
            "matches": [m.to_dict() for m in self.matches],
            "num_matches": len(self.matches)
        }
        if self.inlier_mask is not None:
            result["num_inliers"] = int(np.sum(self.inlier_mask))
        if self.homography is not None:
            result["homography"] = self.homography.tolist()
        return result


class BaseMatcher(ABC):
    """Base class for all feature matchers."""
    
    name: str = "BaseMatcher"
    description: str = "Base matcher class"
    
    @abstractmethod
    def match(
        self,
        descriptors1: np.ndarray,
        descriptors2: np.ndarray,
        norm_type: str = "HAMMING"
    ) -> list[Match]:
        """
        Match descriptors from two images.
        
        Args:
            descriptors1: Descriptors from first image
            descriptors2: Descriptors from second image
            norm_type: Norm type for distance calculation ('L2' or 'HAMMING')
            
        Returns:
            List of Match objects
        """
        pass
    
    def get_config(self) -> dict:
        """Return current configuration."""
        return {}

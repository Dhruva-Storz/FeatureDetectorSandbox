"""Base classes for feature detectors."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple


@dataclass
class KeyPoint:
    """Represents a detected keypoint."""
    x: float
    y: float
    size: float
    angle: float
    response: float
    octave: int
    class_id: int = -1
    
    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "size": self.size,
            "angle": self.angle,
            "response": self.response,
            "octave": self.octave,
            "class_id": self.class_id
        }
    
    @classmethod
    def from_cv2(cls, kp) -> "KeyPoint":
        """Create KeyPoint from cv2.KeyPoint."""
        return cls(
            x=kp.pt[0],
            y=kp.pt[1],
            size=kp.size,
            angle=kp.angle,
            response=kp.response,
            octave=kp.octave,
            class_id=kp.class_id
        )


@dataclass 
class DetectionResult:
    """Result of feature detection."""
    keypoints: list[KeyPoint]
    descriptors: Optional[np.ndarray]
    
    def to_dict(self) -> dict:
        return {
            "keypoints": [kp.to_dict() for kp in self.keypoints],
            "num_keypoints": len(self.keypoints),
            "descriptor_shape": list(self.descriptors.shape) if self.descriptors is not None else None
        }


class BaseDetector(ABC):
    """Base class for all feature detectors."""
    
    name: str = "BaseDetector"
    description: str = "Base detector class"
    
    @abstractmethod
    def detect_and_compute(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> DetectionResult:
        """
        Detect keypoints and compute descriptors.
        
        Args:
            image: Input image (BGR or grayscale)
            mask: Optional mask for detection region
            
        Returns:
            DetectionResult containing keypoints and descriptors
        """
        pass
    
    def get_norm_type(self) -> str:
        """Return the norm type for matching ('L2' or 'HAMMING')."""
        return "HAMMING"
    
    def get_config(self) -> dict:
        """Return current configuration."""
        return {}

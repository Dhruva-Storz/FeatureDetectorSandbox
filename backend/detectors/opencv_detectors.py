"""OpenCV-based feature detectors."""
import cv2
import numpy as np
from typing import Optional

from .base import BaseDetector, DetectionResult, KeyPoint


class ORBDetector(BaseDetector):
    """ORB (Oriented FAST and Rotated BRIEF) detector."""
    
    name = "ORB"
    description = "ORB - Fast binary descriptor, good for real-time applications"
    
    def __init__(self, n_features: int = 3000, scale_factor: float = 1.2, n_levels: int = 8):
        self.n_features = n_features
        self.scale_factor = scale_factor
        self.n_levels = n_levels
        self._detector = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels
        )
    
    def detect_and_compute(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> DetectionResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        kps, descs = self._detector.detectAndCompute(gray, mask)
        
        keypoints = [KeyPoint.from_cv2(kp) for kp in kps] if kps else []
        return DetectionResult(keypoints=keypoints, descriptors=descs)
    
    def get_norm_type(self) -> str:
        return "HAMMING"
    
    def get_config(self) -> dict:
        return {
            "n_features": self.n_features,
            "scale_factor": self.scale_factor,
            "n_levels": self.n_levels
        }


class AKAZEDetector(BaseDetector):
    """AKAZE detector - robust to noise and scale changes."""
    
    name = "AKAZE"
    description = "AKAZE - Robust nonlinear scale space detector"
    
    def __init__(self, threshold: float = 0.001, n_octaves: int = 4, n_octave_layers: int = 4):
        self.threshold = threshold
        self.n_octaves = n_octaves
        self.n_octave_layers = n_octave_layers
        self._detector = cv2.AKAZE_create(
            threshold=threshold,
            nOctaves=n_octaves,
            nOctaveLayers=n_octave_layers
        )
    
    def detect_and_compute(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> DetectionResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        kps, descs = self._detector.detectAndCompute(gray, mask)
        
        keypoints = [KeyPoint.from_cv2(kp) for kp in kps] if kps else []
        return DetectionResult(keypoints=keypoints, descriptors=descs)
    
    def get_norm_type(self) -> str:
        return "HAMMING"
    
    def get_config(self) -> dict:
        return {
            "threshold": self.threshold,
            "n_octaves": self.n_octaves,
            "n_octave_layers": self.n_octave_layers
        }


class BRISKDetector(BaseDetector):
    """BRISK detector - Binary Robust Invariant Scalable Keypoints."""
    
    name = "BRISK"
    description = "BRISK - Fast binary descriptor with scale invariance"
    
    def __init__(self, threshold: int = 30, octaves: int = 3, pattern_scale: float = 1.0):
        self.threshold = threshold
        self.octaves = octaves
        self.pattern_scale = pattern_scale
        self._detector = cv2.BRISK_create(
            thresh=threshold,
            octaves=octaves,
            patternScale=pattern_scale
        )
    
    def detect_and_compute(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> DetectionResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        kps, descs = self._detector.detectAndCompute(gray, mask)
        
        keypoints = [KeyPoint.from_cv2(kp) for kp in kps] if kps else []
        return DetectionResult(keypoints=keypoints, descriptors=descs)
    
    def get_norm_type(self) -> str:
        return "HAMMING"
    
    def get_config(self) -> dict:
        return {
            "threshold": self.threshold,
            "octaves": self.octaves,
            "pattern_scale": self.pattern_scale
        }


class SIFTDetector(BaseDetector):
    """SIFT detector - Scale-Invariant Feature Transform."""

    name = "SIFT"
    description = "SIFT - Classic scale and rotation invariant detector"

    def __init__(self, n_features: int = 0, n_octave_layers: int = 3,
                 contrast_threshold: float = 0.04, edge_threshold: float = 10.0,
                 sigma: float = 1.6):
        self.n_features = n_features
        self.n_octave_layers = n_octave_layers
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma

        # Check if SIFT is available
        if not hasattr(cv2, 'SIFT_create'):
            raise RuntimeError("SIFT not available in this OpenCV build")

        self._detector = cv2.SIFT_create(
            nfeatures=n_features,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma
        )

    def detect_and_compute(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> DetectionResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        kps, descs = self._detector.detectAndCompute(gray, mask)

        keypoints = [KeyPoint.from_cv2(kp) for kp in kps] if kps else []
        return DetectionResult(keypoints=keypoints, descriptors=descs)

    def get_norm_type(self) -> str:
        return "L2"

    def get_config(self) -> dict:
        return {
            "n_features": self.n_features,
            "n_octave_layers": self.n_octave_layers,
            "contrast_threshold": self.contrast_threshold,
            "edge_threshold": self.edge_threshold,
            "sigma": self.sigma
        }


class SURFDetector(BaseDetector):
    """SURF detector - Speeded-Up Robust Features (patent expired 2020)."""

    name = "SURF"
    description = "SURF - Fast approximation of SIFT using box filters"

    def __init__(self, hessian_threshold: float = 400.0, n_octaves: int = 4,
                 n_octave_layers: int = 3, extended: bool = False, upright: bool = False):
        self.hessian_threshold = hessian_threshold
        self.n_octaves = n_octaves
        self.n_octave_layers = n_octave_layers
        self.extended = extended
        self.upright = upright

        # SURF requires opencv-contrib-python
        if not hasattr(cv2, 'xfeatures2d') or not hasattr(cv2.xfeatures2d, 'SURF_create'):
            raise RuntimeError(
                "SURF not available. Install opencv-contrib-python: "
                "pip install opencv-contrib-python"
            )

        self._detector = cv2.xfeatures2d.SURF_create(
            hessianThreshold=hessian_threshold,
            nOctaves=n_octaves,
            nOctaveLayers=n_octave_layers,
            extended=extended,
            upright=upright
        )

    def detect_and_compute(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> DetectionResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        kps, descs = self._detector.detectAndCompute(gray, mask)

        keypoints = [KeyPoint.from_cv2(kp) for kp in kps] if kps else []
        return DetectionResult(keypoints=keypoints, descriptors=descs)

    def get_norm_type(self) -> str:
        return "L2"

    def get_config(self) -> dict:
        return {
            "hessian_threshold": self.hessian_threshold,
            "n_octaves": self.n_octaves,
            "n_octave_layers": self.n_octave_layers,
            "extended": self.extended,
            "upright": self.upright
        }

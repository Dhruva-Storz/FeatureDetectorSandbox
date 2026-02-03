"""Robust estimation methods (RANSAC, MAGSAC++, etc.)."""
import cv2
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class RobustEstimationResult:
    """Result of robust estimation."""
    homography: Optional[np.ndarray]
    inlier_mask: np.ndarray
    num_inliers: int
    
    def to_dict(self) -> dict:
        return {
            "homography": self.homography.tolist() if self.homography is not None else None,
            "inlier_mask": self.inlier_mask.tolist(),
            "num_inliers": self.num_inliers
        }


# Available robust methods
ROBUST_METHODS = {
    "RANSAC": cv2.RANSAC,
    "LMEDS": cv2.LMEDS,
    "RHO": cv2.RHO,
}

# Check for USAC methods (OpenCV 4.5+)
if hasattr(cv2, 'USAC_MAGSAC'):
    ROBUST_METHODS["MAGSAC++"] = cv2.USAC_MAGSAC
if hasattr(cv2, 'USAC_DEFAULT'):
    ROBUST_METHODS["USAC_DEFAULT"] = cv2.USAC_DEFAULT
if hasattr(cv2, 'USAC_ACCURATE'):
    ROBUST_METHODS["USAC_ACCURATE"] = cv2.USAC_ACCURATE


def find_homography_robust(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    method: str = "RANSAC",
    reproj_threshold: float = 5.0,
    max_iters: int = 2000,
    confidence: float = 0.995
) -> RobustEstimationResult:
    """
    Find homography using robust estimation.
    
    Args:
        src_points: Source points (Nx2)
        dst_points: Destination points (Nx2)
        method: Robust method name
        reproj_threshold: Reprojection error threshold
        max_iters: Maximum iterations
        confidence: Confidence level
        
    Returns:
        RobustEstimationResult with homography and inlier mask
    """
    if method not in ROBUST_METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {list(ROBUST_METHODS.keys())}")
    
    cv_method = ROBUST_METHODS[method]
    
    # Ensure correct shape
    src_pts = src_points.reshape(-1, 1, 2).astype(np.float32)
    dst_pts = dst_points.reshape(-1, 1, 2).astype(np.float32)
    
    H, mask = cv2.findHomography(
        src_pts, dst_pts,
        method=cv_method,
        ransacReprojThreshold=reproj_threshold,
        maxIters=max_iters,
        confidence=confidence
    )
    
    if mask is None:
        mask = np.zeros(len(src_points), dtype=np.uint8)
    else:
        mask = mask.ravel()
    
    return RobustEstimationResult(
        homography=H,
        inlier_mask=mask,
        num_inliers=int(np.sum(mask))
    )


def list_robust_methods() -> list[dict]:
    """List available robust estimation methods."""
    return [
        {"name": name, "available": True}
        for name in ROBUST_METHODS.keys()
    ]

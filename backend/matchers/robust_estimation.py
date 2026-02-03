"""Robust estimation methods (RANSAC, MAGSAC++, etc.)."""
import cv2
import numpy as np
from typing import Optional, Tuple, Literal
from dataclasses import dataclass


@dataclass
class RobustEstimationResult:
    """Result of robust estimation."""
    homography: Optional[np.ndarray]
    fundamental: Optional[np.ndarray] = None
    inlier_mask: np.ndarray = None
    num_inliers: int = 0
    model_type: str = "homography"
    
    def to_dict(self) -> dict:
        result = {
            "inlier_mask": self.inlier_mask.tolist() if self.inlier_mask is not None else [],
            "num_inliers": self.num_inliers,
            "model_type": self.model_type
        }
        if self.homography is not None:
            result["homography"] = self.homography.tolist()
        if self.fundamental is not None:
            result["fundamental"] = self.fundamental.tolist()
        return result


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
        fundamental=None,
        inlier_mask=mask,
        num_inliers=int(np.sum(mask)),
        model_type="homography"
    )


def list_robust_methods() -> list[dict]:
    """List available robust estimation methods."""
    return [
        {"name": name, "available": True}
        for name in ROBUST_METHODS.keys()
    ]


def find_fundamental_robust(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    method: str = "RANSAC",
    reproj_threshold: float = 3.0,
    max_iters: int = 2000,
    confidence: float = 0.995
) -> RobustEstimationResult:
    """
    Find fundamental matrix using robust estimation.
    Better for stereo pairs than homography as it allows
    non-parallel epipolar geometry (convergent cameras, lens distortion).
    
    Args:
        src_points: Source points (Nx2)
        dst_points: Destination points (Nx2)
        method: Robust method name
        reproj_threshold: Epipolar distance threshold in pixels
        max_iters: Maximum iterations
        confidence: Confidence level
        
    Returns:
        RobustEstimationResult with fundamental matrix and inlier mask
    """
    if method not in ROBUST_METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {list(ROBUST_METHODS.keys())}")
    
    cv_method = ROBUST_METHODS[method]
    
    # Ensure correct shape
    src_pts = src_points.reshape(-1, 1, 2).astype(np.float32)
    dst_pts = dst_points.reshape(-1, 1, 2).astype(np.float32)
    
    # Need at least 8 points for fundamental matrix
    if len(src_pts) < 8:
        return RobustEstimationResult(
            homography=None,
            fundamental=None,
            inlier_mask=np.ones(len(src_points), dtype=np.uint8),
            num_inliers=len(src_points),
            model_type="fundamental"
        )
    
    F, mask = cv2.findFundamentalMat(
        src_pts, dst_pts,
        method=cv_method,
        ransacReprojThreshold=reproj_threshold,
        confidence=confidence,
        maxIters=max_iters
    )
    
    if mask is None:
        mask = np.zeros(len(src_points), dtype=np.uint8)
    else:
        mask = mask.ravel()
    
    return RobustEstimationResult(
        homography=None,
        fundamental=F,
        inlier_mask=mask,
        num_inliers=int(np.sum(mask)),
        model_type="fundamental"
    )


def refine_matches_robust(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    method: str = "RANSAC",
    model_type: Literal["homography", "fundamental"] = "homography",
    reproj_threshold: float = 5.0,
    max_iters: int = 2000,
    confidence: float = 0.995
) -> RobustEstimationResult:
    """
    Unified interface for robust match refinement.
    
    Args:
        src_points: Source points (Nx2)
        dst_points: Destination points (Nx2)
        method: Robust method (RANSAC, MAGSAC++, LMEDS, RHO)
        model_type: "homography" or "fundamental"
        reproj_threshold: Error threshold in pixels
        max_iters: Maximum iterations
        confidence: Confidence level (0-1)
        
    Returns:
        RobustEstimationResult with computed model and inlier mask
    """
    if model_type == "fundamental":
        return find_fundamental_robust(
            src_points, dst_points,
            method=method,
            reproj_threshold=reproj_threshold,
            max_iters=max_iters,
            confidence=confidence
        )
    else:
        return find_homography_robust(
            src_points, dst_points,
            method=method,
            reproj_threshold=reproj_threshold,
            max_iters=max_iters,
            confidence=confidence
        )

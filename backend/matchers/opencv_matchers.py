"""OpenCV-based feature matchers."""
import cv2
import numpy as np
from typing import Optional

from .base import BaseMatcher, Match, MatchResult


class BFMatcher(BaseMatcher):
    """Brute Force matcher with optional ratio test."""
    
    name = "BF"
    description = "Brute Force matcher with KNN and ratio test support"
    
    def __init__(self, cross_check: bool = False):
        self.cross_check = cross_check
    
    def match(
        self,
        descriptors1: np.ndarray,
        descriptors2: np.ndarray,
        norm_type: str = "HAMMING"
    ) -> list[Match]:
        """Simple 1-to-1 matching."""
        cv_norm = cv2.NORM_HAMMING if norm_type == "HAMMING" else cv2.NORM_L2
        matcher = cv2.BFMatcher(cv_norm, crossCheck=True)
        
        cv_matches = matcher.match(descriptors1, descriptors2)
        
        return [
            Match(
                query_idx=m.queryIdx,
                train_idx=m.trainIdx,
                distance=m.distance
            )
            for m in cv_matches
        ]
    
    def knn_match(
        self,
        descriptors1: np.ndarray,
        descriptors2: np.ndarray,
        k: int = 2,
        norm_type: str = "HAMMING",
        ratio_threshold: Optional[float] = None
    ) -> list[Match]:
        """
        KNN matching with optional Lowe's ratio test.
        
        Args:
            descriptors1: First set of descriptors
            descriptors2: Second set of descriptors
            k: Number of nearest neighbors
            norm_type: Distance norm type
            ratio_threshold: Lowe's ratio threshold (e.g., 0.75). If None, all matches returned.
            
        Returns:
            List of filtered matches
        """
        cv_norm = cv2.NORM_HAMMING if norm_type == "HAMMING" else cv2.NORM_L2
        matcher = cv2.BFMatcher(cv_norm, crossCheck=False)
        
        knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=k)
        
        matches = []
        for match_group in knn_matches:
            if len(match_group) < 1:
                continue
            
            best = match_group[0]
            
            # Apply ratio test if threshold provided and we have 2+ matches
            if ratio_threshold is not None and len(match_group) >= 2:
                second = match_group[1]
                if best.distance >= second.distance * ratio_threshold:
                    continue
            
            matches.append(Match(
                query_idx=best.queryIdx,
                train_idx=best.trainIdx,
                distance=best.distance
            ))
        
        return matches
    
    def get_config(self) -> dict:
        return {"cross_check": self.cross_check}


def filter_by_distance(
    matches: list[Match],
    keypoints1: list,
    keypoints2: list,
    max_distance: float
) -> list[Match]:
    """
    Filter matches by spatial distance between matched keypoints.
    
    Args:
        matches: List of matches to filter
        keypoints1: Keypoints from first image
        keypoints2: Keypoints from second image
        max_distance: Maximum allowed spatial distance in pixels
        
    Returns:
        Filtered list of matches
    """
    filtered = []
    for m in matches:
        kp1 = keypoints1[m.query_idx]
        kp2 = keypoints2[m.train_idx]
        
        dx = kp1.x - kp2.x
        dy = kp1.y - kp2.y
        dist = np.sqrt(dx*dx + dy*dy)
        
        if dist <= max_distance:
            filtered.append(m)
    
    return filtered


def filter_by_rank(
    matches: list[Match],
    rank_start: int = 0,
    rank_end: int = 100
) -> list[Match]:
    """
    Filter matches by rank (assumes matches are sorted by quality).
    
    Args:
        matches: List of matches (should be sorted by distance)
        rank_start: Start index (inclusive)
        rank_end: End index (exclusive)
        
    Returns:
        Sliced list of matches
    """
    sorted_matches = sorted(matches, key=lambda m: m.distance)
    return sorted_matches[max(0, rank_start):min(len(sorted_matches), rank_end)]

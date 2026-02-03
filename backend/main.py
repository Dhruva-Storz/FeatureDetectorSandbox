"""
Feature Detector Testbed - FastAPI Backend

This provides a REST API for testing various feature detection and matching algorithms.
Designed to be extensible for adding advanced models like LoFTR, SuperPoint+LightGlue.
"""
import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import os


def get_gpu_memory_info() -> dict:
    """Get GPU memory usage info if torch/CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            reserved = torch.cuda.memory_reserved() / 1024**2  # MB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
            return {
                "allocated_mb": round(allocated, 1),
                "reserved_mb": round(reserved, 1),
                "total_mb": round(total, 1),
                "free_mb": round(total - reserved, 1),
            }
    except Exception:
        pass
    return None

from .detectors import get_detector, list_detectors, AVAILABLE_DETECTORS
from .matchers import get_matcher, list_matchers
from .matchers.opencv_matchers import filter_by_distance, filter_by_rank
from .matchers.robust_estimation import find_homography_robust, list_robust_methods
from .dense_matchers import get_dense_matcher, list_dense_matchers, AVAILABLE_DENSE_MATCHERS
from .utils import (
    decode_image_from_base64,
    encode_image_to_base64,
    center_image,
    draw_matches_side_by_side,
    draw_matches_anaglyph,
    draw_dense_matches_side_by_side,
    draw_dense_matches_anaglyph,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Feature Detector Testbed",
    description="API for testing stereo feature detection and matching algorithms",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ============== Request/Response Models ==============

class MatchRequest(BaseModel):
    """Request model for feature matching."""
    image1: str = Field(..., description="Base64 encoded left image")
    image2: str = Field(..., description="Base64 encoded right image")
    detector: str = Field(default="ORB", description="Detector name (ORB, AKAZE, BRISK, SIFT)")
    
    # Matching parameters
    use_ratio_test: bool = Field(default=False, description="Use Lowe's ratio test")
    ratio_threshold: float = Field(default=0.75, ge=0.1, le=0.99, description="Ratio test threshold")
    
    # Filtering parameters
    rank_start: int = Field(default=0, ge=0, description="Start rank for filtering")
    rank_end: int = Field(default=100, ge=1, description="End rank for filtering")
    max_spatial_distance: Optional[float] = Field(default=None, description="Max spatial distance in pixels")
    
    # Visualization
    viz_mode: str = Field(default="sidebyside", description="Visualization mode: 'sidebyside' or 'anaglyph'")


class RefineRequest(BaseModel):
    """Request model for robust refinement."""
    image1: str = Field(..., description="Base64 encoded left image")
    image2: str = Field(..., description="Base64 encoded right image")
    keypoints1: list[dict] = Field(..., description="Keypoints from first image")
    keypoints2: list[dict] = Field(..., description="Keypoints from second image")
    matches: list[dict] = Field(..., description="Current matches to refine")
    method: str = Field(default="RANSAC", description="Robust method")
    reproj_threshold: float = Field(default=5.0, description="Reprojection threshold")
    viz_mode: str = Field(default="sidebyside", description="Visualization mode")


class TimingInfo(BaseModel):
    """Timing information for operations."""
    total_ms: float = 0.0
    detection_ms: Optional[float] = None
    matching_ms: Optional[float] = None
    refinement_ms: Optional[float] = None
    visualization_ms: Optional[float] = None


class GPUMemoryInfo(BaseModel):
    """GPU memory usage information."""
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    total_mb: float = 0.0
    free_mb: float = 0.0


class MatchResponse(BaseModel):
    """Response model for matching operations."""
    success: bool
    message: str
    result_image: Optional[str] = None
    num_candidates: int = 0
    num_matches: int = 0
    detector_used: str = ""
    keypoints1: Optional[list[dict]] = None
    keypoints2: Optional[list[dict]] = None
    matches: Optional[list[dict]] = None
    timing: Optional[TimingInfo] = None


class RefineResponse(BaseModel):
    """Response model for refinement operations."""
    success: bool
    message: str
    result_image: Optional[str] = None
    num_inliers: int = 0
    num_total: int = 0
    method_used: str = ""
    matches: Optional[list[dict]] = None
    timing: Optional[TimingInfo] = None


class DenseMatchRequest(BaseModel):
    """Request model for dense/end-to-end matching."""
    image1: str = Field(..., description="Base64 encoded left image")
    image2: str = Field(..., description="Base64 encoded right image")
    matcher: str = Field(default="SuperPoint+LightGlue", description="Dense matcher name (SuperPoint+LightGlue, DISK+LightGlue, EfficientLoFTR)")

    # Matching parameters
    threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence threshold for matches")

    # Filtering parameters
    rank_start: int = Field(default=0, ge=0, description="Start rank for filtering")
    rank_end: int = Field(default=500, ge=1, description="End rank for filtering")

    # Visualization
    viz_mode: str = Field(default="sidebyside", description="Visualization mode: 'sidebyside' or 'anaglyph'")


class DenseMatchResponse(BaseModel):
    """Response model for dense matching operations."""
    success: bool
    message: str
    result_image: Optional[str] = None
    num_matches: int = 0
    matcher_used: str = ""
    keypoints1: Optional[list[list[float]]] = None
    keypoints2: Optional[list[list[float]]] = None
    scores: Optional[list[float]] = None
    timing: Optional[TimingInfo] = None
    gpu_memory: Optional[GPUMemoryInfo] = None


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """Serve the frontend."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Feature Detector Testbed API", "docs": "/docs"}


@app.get("/api/detectors")
async def get_available_detectors():
    """List all available feature detectors."""
    return {"detectors": list_detectors()}


@app.get("/api/matchers")
async def get_available_matchers():
    """List all available matchers."""
    return {"matchers": list_matchers()}


@app.get("/api/robust-methods")
async def get_robust_methods():
    """List all available robust estimation methods."""
    return {"methods": list_robust_methods()}


@app.get("/api/dense-matchers")
async def get_available_dense_matchers():
    """List all available dense/end-to-end matchers (LightGlue, EfficientLoFTR, etc.)."""
    return {"matchers": list_dense_matchers()}


@app.post("/api/match", response_model=MatchResponse)
async def compute_matches(request: MatchRequest):
    """
    Compute feature matches between two images.

    This endpoint:
    1. Decodes the input images
    2. Runs feature detection with the specified detector
    3. Matches features with optional ratio test
    4. Filters by rank and spatial distance
    5. Returns visualization and match data
    """
    try:
        total_start = time.perf_counter()
        logger.info(f"Computing matches with detector: {request.detector}")

        # Decode images
        img1 = decode_image_from_base64(request.image1)
        img2 = decode_image_from_base64(request.image2)

        logger.info(f"Image 1 shape: {img1.shape}, Image 2 shape: {img2.shape}")

        # Center images to common size
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])

        img1_centered, x1_off, y1_off = center_image(img1, w, h)
        img2_centered, x2_off, y2_off = center_image(img2, w, h)

        # Get detector
        try:
            detector = get_detector(request.detector)
        except Exception as e:
            logger.warning(f"Detector {request.detector} failed: {e}, falling back to AKAZE")
            detector = get_detector("AKAZE")

        # Detect features (timed)
        detect_start = time.perf_counter()
        result1 = detector.detect_and_compute(img1_centered)
        result2 = detector.detect_and_compute(img2_centered)
        detection_ms = (time.perf_counter() - detect_start) * 1000

        logger.info(f"Detected {len(result1.keypoints)} and {len(result2.keypoints)} keypoints in {detection_ms:.1f}ms")

        if result1.descriptors is None or result2.descriptors is None:
            return MatchResponse(
                success=False,
                message="No descriptors found in one or both images",
                detector_used=detector.name
            )

        # Match features (timed)
        match_start = time.perf_counter()
        matcher = get_matcher("BF")
        norm_type = detector.get_norm_type()

        if request.use_ratio_test:
            matches = matcher.knn_match(
                result1.descriptors,
                result2.descriptors,
                k=2,
                norm_type=norm_type,
                ratio_threshold=request.ratio_threshold
            )
        else:
            matches = matcher.knn_match(
                result1.descriptors,
                result2.descriptors,
                k=2,
                norm_type=norm_type,
                ratio_threshold=None
            )
        matching_ms = (time.perf_counter() - match_start) * 1000

        num_candidates = len(matches)
        logger.info(f"Found {num_candidates} candidates in {matching_ms:.1f}ms")

        # Filter by rank
        matches = filter_by_rank(matches, request.rank_start, request.rank_end)

        # Filter by spatial distance
        if request.max_spatial_distance is not None and request.max_spatial_distance < 2000:
            matches = filter_by_distance(
                matches,
                result1.keypoints,
                result2.keypoints,
                request.max_spatial_distance
            )

        logger.info(f"After filtering: {len(matches)} matches")

        # Draw visualization (timed)
        viz_start = time.perf_counter()
        if request.viz_mode == "anaglyph":
            result_img = draw_matches_anaglyph(
                img1_centered, img2_centered,
                result1.keypoints, result2.keypoints,
                matches
            )
        else:
            result_img = draw_matches_side_by_side(
                img1_centered, img2_centered,
                result1.keypoints, result2.keypoints,
                matches
            )
        visualization_ms = (time.perf_counter() - viz_start) * 1000

        result_b64 = encode_image_to_base64(result_img)
        total_ms = (time.perf_counter() - total_start) * 1000

        timing = TimingInfo(
            total_ms=round(total_ms, 1),
            detection_ms=round(detection_ms, 1),
            matching_ms=round(matching_ms, 1),
            visualization_ms=round(visualization_ms, 1),
        )

        return MatchResponse(
            success=True,
            message=f"{detector.name} | {len(matches)} matches | {total_ms:.0f}ms",
            result_image=result_b64,
            num_candidates=num_candidates,
            num_matches=len(matches),
            detector_used=detector.name,
            keypoints1=[kp.to_dict() for kp in result1.keypoints],
            keypoints2=[kp.to_dict() for kp in result2.keypoints],
            matches=[m.to_dict() for m in matches],
            timing=timing,
        )

    except Exception as e:
        logger.exception("Error in compute_matches")
        return MatchResponse(
            success=False,
            message=f"Error: {str(e)}"
        )


@app.post("/api/refine", response_model=RefineResponse)
async def refine_matches(request: RefineRequest):
    """
    Refine matches using robust estimation (RANSAC, MAGSAC++, etc.).
    """
    try:
        total_start = time.perf_counter()
        logger.info(f"Refining with method: {request.method}")

        if len(request.matches) < 4:
            return RefineResponse(
                success=False,
                message="Need at least 4 matches for robust estimation"
            )

        # Decode images
        img1 = decode_image_from_base64(request.image1)
        img2 = decode_image_from_base64(request.image2)

        # Center images
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img1_centered, _, _ = center_image(img1, w, h)
        img2_centered, _, _ = center_image(img2, w, h)

        # Convert keypoints back to our format
        from .detectors.base import KeyPoint
        keypoints1 = [KeyPoint(**kp) for kp in request.keypoints1]
        keypoints2 = [KeyPoint(**kp) for kp in request.keypoints2]

        # Extract point correspondences
        src_points = []
        dst_points = []

        for m in request.matches:
            kp1 = keypoints1[m["query_idx"]]
            kp2 = keypoints2[m["train_idx"]]
            src_points.append([kp1.x, kp1.y])
            dst_points.append([kp2.x, kp2.y])

        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)

        # Run robust estimation (timed)
        refine_start = time.perf_counter()
        result = find_homography_robust(
            src_points, dst_points,
            method=request.method,
            reproj_threshold=request.reproj_threshold
        )
        refinement_ms = (time.perf_counter() - refine_start) * 1000

        # Filter matches by inlier mask
        from .matchers.base import Match
        inlier_matches = []
        for i, m in enumerate(request.matches):
            if result.inlier_mask[i] == 1:
                inlier_matches.append(Match(
                    query_idx=m["query_idx"],
                    train_idx=m["train_idx"],
                    distance=m["distance"]
                ))

        logger.info(f"Inliers: {len(inlier_matches)}/{len(request.matches)} in {refinement_ms:.1f}ms")

        # Draw visualization (timed)
        viz_start = time.perf_counter()
        if request.viz_mode == "anaglyph":
            result_img = draw_matches_anaglyph(
                img1_centered, img2_centered,
                keypoints1, keypoints2,
                inlier_matches
            )
        else:
            result_img = draw_matches_side_by_side(
                img1_centered, img2_centered,
                keypoints1, keypoints2,
                inlier_matches
            )
        visualization_ms = (time.perf_counter() - viz_start) * 1000

        result_b64 = encode_image_to_base64(result_img)
        total_ms = (time.perf_counter() - total_start) * 1000

        timing = TimingInfo(
            total_ms=round(total_ms, 1),
            refinement_ms=round(refinement_ms, 1),
            visualization_ms=round(visualization_ms, 1),
        )

        return RefineResponse(
            success=True,
            message=f"{request.method}: {len(inlier_matches)}/{len(request.matches)} inliers | {refinement_ms:.0f}ms",
            result_image=result_b64,
            num_inliers=len(inlier_matches),
            num_total=len(request.matches),
            method_used=request.method,
            matches=[m.to_dict() for m in inlier_matches],
            timing=timing,
        )

    except Exception as e:
        logger.exception("Error in refine_matches")
        return RefineResponse(
            success=False,
            message=f"Error: {str(e)}"
        )


@app.post("/api/dense-match", response_model=DenseMatchResponse)
async def compute_dense_matches(request: DenseMatchRequest):
    """
    Compute matches using a dense/end-to-end matcher (LightGlue, EfficientLoFTR).

    These matchers combine detection and matching in one step, often producing
    higher quality matches than traditional detect-then-match pipelines.
    """
    try:
        total_start = time.perf_counter()
        logger.info(f"Computing dense matches with matcher: {request.matcher}")

        # Check if dense matchers are available
        if not AVAILABLE_DENSE_MATCHERS:
            return DenseMatchResponse(
                success=False,
                message="No dense matchers available. Install: pip install torch transformers"
            )

        # Decode images
        img1 = decode_image_from_base64(request.image1)
        img2 = decode_image_from_base64(request.image2)

        logger.info(f"Image 1 shape: {img1.shape}, Image 2 shape: {img2.shape}")

        # Center images to common size
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])

        img1_centered, _, _ = center_image(img1, w, h)
        img2_centered, _, _ = center_image(img2, w, h)

        # Get dense matcher
        try:
            matcher = get_dense_matcher(request.matcher)
        except Exception as e:
            return DenseMatchResponse(
                success=False,
                message=f"Failed to load matcher {request.matcher}: {str(e)}"
            )

        # Run matching (timed)
        match_start = time.perf_counter()
        result = matcher.match(img1_centered, img2_centered, threshold=request.threshold)
        matching_ms = (time.perf_counter() - match_start) * 1000

        # Get GPU memory after inference
        gpu_mem = get_gpu_memory_info()

        logger.info(f"Found {len(result.scores)} matches in {matching_ms:.1f}ms")

        # Filter by rank if needed
        if request.rank_start > 0 or request.rank_end < len(result.scores):
            result = result.filter_by_rank(request.rank_start, request.rank_end)

        logger.info(f"After filtering: {len(result.scores)} matches")

        # Draw visualization (timed)
        viz_start = time.perf_counter()
        if request.viz_mode == "anaglyph":
            result_img = draw_dense_matches_anaglyph(
                img1_centered, img2_centered,
                result.keypoints1, result.keypoints2,
                result.scores
            )
        else:
            result_img = draw_dense_matches_side_by_side(
                img1_centered, img2_centered,
                result.keypoints1, result.keypoints2,
                result.scores
            )
        visualization_ms = (time.perf_counter() - viz_start) * 1000

        result_b64 = encode_image_to_base64(result_img)
        total_ms = (time.perf_counter() - total_start) * 1000

        timing = TimingInfo(
            total_ms=round(total_ms, 1),
            matching_ms=round(matching_ms, 1),
            visualization_ms=round(visualization_ms, 1),
        )

        gpu_memory = None
        if gpu_mem:
            gpu_memory = GPUMemoryInfo(**gpu_mem)

        return DenseMatchResponse(
            success=True,
            message=f"{matcher.name} | {len(result.scores)} matches | {matching_ms:.0f}ms",
            result_image=result_b64,
            num_matches=len(result.scores),
            matcher_used=matcher.name,
            keypoints1=result.keypoints1.tolist(),
            keypoints2=result.keypoints2.tolist(),
            scores=result.scores.tolist(),
            timing=timing,
            gpu_memory=gpu_memory,
        )

    except Exception as e:
        logger.exception("Error in compute_dense_matches")
        return DenseMatchResponse(
            success=False,
            message=f"Error: {str(e)}"
        )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}

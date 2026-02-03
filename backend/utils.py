"""Utility functions for image processing."""
import cv2
import numpy as np
import base64
from io import BytesIO
from typing import Tuple, Optional


def decode_image_from_base64(data: str) -> np.ndarray:
    """
    Decode a base64 encoded image to numpy array.
    
    Args:
        data: Base64 encoded image string (may include data URI prefix)
        
    Returns:
        Image as numpy array (BGR)
    """
    # Remove data URI prefix if present
    if ',' in data:
        data = data.split(',', 1)[1]
    
    img_bytes = base64.b64decode(data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image")
    
    return img


def encode_image_to_base64(img: np.ndarray, format: str = ".png") -> str:
    """
    Encode a numpy array image to base64 string.
    
    Args:
        img: Image as numpy array (BGR)
        format: Image format (.png, .jpg, etc.)
        
    Returns:
        Base64 encoded string with data URI prefix
    """
    success, buffer = cv2.imencode(format, img)
    if not success:
        raise ValueError("Failed to encode image")
    
    b64 = base64.b64encode(buffer).decode('utf-8')
    mime_type = "image/png" if format == ".png" else "image/jpeg"
    return f"data:{mime_type};base64,{b64}"


def center_image(img: np.ndarray, target_w: int, target_h: int) -> Tuple[np.ndarray, int, int]:
    """
    Center an image in a canvas of specified size.
    
    Args:
        img: Input image
        target_w: Target width
        target_h: Target height
        
    Returns:
        Tuple of (centered image, x_offset, y_offset)
    """
    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1
    
    if channels == 1:
        canvas = np.zeros((target_h, target_w), dtype=img.dtype)
    else:
        canvas = np.zeros((target_h, target_w, channels), dtype=img.dtype)
    
    x_off = (target_w - w) // 2
    y_off = (target_h - h) // 2
    
    canvas[y_off:y_off+h, x_off:x_off+w] = img
    
    return canvas, x_off, y_off


def draw_matches_side_by_side(
    img1: np.ndarray,
    img2: np.ndarray,
    keypoints1: list,
    keypoints2: list,
    matches: list,
    match_color: Tuple[int, int, int] = (0, 255, 0),
    line_thickness: int = 1
) -> np.ndarray:
    """
    Draw matches between two images side by side.
    
    Args:
        img1: First image
        img2: Second image
        keypoints1: Keypoints from first image
        keypoints2: Keypoints from second image
        matches: List of Match objects
        match_color: Color for match lines (BGR)
        line_thickness: Thickness of match lines
        
    Returns:
        Combined image with matches drawn
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create combined image
    h = max(h1, h2)
    w = w1 + w2
    
    if len(img1.shape) == 2:
        combined = np.zeros((h, w), dtype=img1.dtype)
    else:
        combined = np.zeros((h, w, img1.shape[2]), dtype=img1.dtype)
    
    combined[:h1, :w1] = img1
    combined[:h2, w1:w1+w2] = img2
    
    # Ensure we have a color image for drawing
    if len(combined.shape) == 2:
        combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    
    # Draw matches
    for m in matches:
        kp1 = keypoints1[m.query_idx]
        kp2 = keypoints2[m.train_idx]
        
        pt1 = (int(kp1.x), int(kp1.y))
        pt2 = (int(kp2.x + w1), int(kp2.y))
        
        cv2.line(combined, pt1, pt2, match_color, line_thickness, cv2.LINE_AA)
        cv2.circle(combined, pt1, 3, match_color, -1)
        cv2.circle(combined, pt2, 3, match_color, -1)
    
    return combined


def draw_dense_matches_side_by_side(
    img1: np.ndarray,
    img2: np.ndarray,
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    scores: Optional[np.ndarray] = None,
    match_color: Tuple[int, int, int] = (0, 255, 0),
    line_thickness: int = 1
) -> np.ndarray:
    """
    Draw dense matches between two images side by side.

    Args:
        img1: First image
        img2: Second image
        keypoints1: Keypoint coordinates from first image (Nx2)
        keypoints2: Keypoint coordinates from second image (Nx2)
        scores: Optional match scores for color coding
        match_color: Color for match lines (BGR)
        line_thickness: Thickness of match lines

    Returns:
        Combined image with matches drawn
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Create combined image
    h = max(h1, h2)
    w = w1 + w2

    if len(img1.shape) == 2:
        combined = np.zeros((h, w), dtype=img1.dtype)
    else:
        combined = np.zeros((h, w, img1.shape[2]), dtype=img1.dtype)

    combined[:h1, :w1] = img1
    combined[:h2, w1:w1+w2] = img2

    # Ensure we have a color image for drawing
    if len(combined.shape) == 2:
        combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)

    # Draw matches
    for i in range(len(keypoints1)):
        pt1 = (int(keypoints1[i, 0]), int(keypoints1[i, 1]))
        pt2 = (int(keypoints2[i, 0] + w1), int(keypoints2[i, 1]))

        # Color code by score if available
        if scores is not None:
            # Green to red based on score
            score = float(scores[i])
            color = (0, int(255 * score), int(255 * (1 - score)))
        else:
            color = match_color

        cv2.line(combined, pt1, pt2, color, line_thickness, cv2.LINE_AA)
        cv2.circle(combined, pt1, 3, color, -1)
        cv2.circle(combined, pt2, 3, color, -1)

    return combined


def draw_dense_matches_anaglyph(
    img1: np.ndarray,
    img2: np.ndarray,
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    scores: Optional[np.ndarray] = None,
    line_color: Tuple[int, int, int] = (0, 255, 255),  # Yellow
    kp1_color: Tuple[int, int, int] = (0, 0, 255),     # Red
    kp2_color: Tuple[int, int, int] = (255, 255, 0),   # Cyan
    line_thickness: int = 1
) -> np.ndarray:
    """
    Create an anaglyph visualization of dense matches.

    Args:
        img1: Left image
        img2: Right image
        keypoints1: Keypoint coordinates from left image (Nx2)
        keypoints2: Keypoint coordinates from right image (Nx2)
        scores: Optional match scores
        line_color: Color for match lines (BGR)
        kp1_color: Color for left keypoints (BGR)
        kp2_color: Color for right keypoints (BGR)
        line_thickness: Thickness of match lines

    Returns:
        Anaglyph image with matches
    """
    # Ensure both images are the same size
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    w = max(w1, w2)

    # Center images
    img1_centered, _, _ = center_image(img1, w, h)
    img2_centered, _, _ = center_image(img2, w, h)

    # Ensure color images
    if len(img1_centered.shape) == 2:
        img1_centered = cv2.cvtColor(img1_centered, cv2.COLOR_GRAY2BGR)
    if len(img2_centered.shape) == 2:
        img2_centered = cv2.cvtColor(img2_centered, cv2.COLOR_GRAY2BGR)

    # Create anaglyph (Red from left, Green+Blue from right)
    b1, g1, r1 = cv2.split(img1_centered)
    b2, g2, r2 = cv2.split(img2_centered)

    anaglyph = cv2.merge([b2, g2, r1])

    # Draw matches
    for i in range(len(keypoints1)):
        pt1 = (int(keypoints1[i, 0]), int(keypoints1[i, 1]))
        pt2 = (int(keypoints2[i, 0]), int(keypoints2[i, 1]))

        # Color code by score if available
        if scores is not None:
            score = float(scores[i])
            color = (0, int(255 * score), int(255 * (1 - score)))
        else:
            color = line_color

        cv2.line(anaglyph, pt1, pt2, color, line_thickness, cv2.LINE_AA)
        cv2.circle(anaglyph, pt1, 3, kp1_color, 1)
        cv2.circle(anaglyph, pt2, 3, kp2_color, 1)

    return anaglyph


def draw_matches_anaglyph(
    img1: np.ndarray,
    img2: np.ndarray,
    keypoints1: list,
    keypoints2: list,
    matches: list,
    line_color: Tuple[int, int, int] = (0, 255, 255),  # Yellow
    kp1_color: Tuple[int, int, int] = (0, 0, 255),     # Red
    kp2_color: Tuple[int, int, int] = (255, 255, 0),   # Cyan
    line_thickness: int = 1
) -> np.ndarray:
    """
    Create an anaglyph visualization of matches.
    
    Args:
        img1: Left image
        img2: Right image  
        keypoints1: Keypoints from left image
        keypoints2: Keypoints from right image
        matches: List of Match objects
        line_color: Color for match lines (BGR)
        kp1_color: Color for left keypoints (BGR)
        kp2_color: Color for right keypoints (BGR)
        line_thickness: Thickness of match lines
        
    Returns:
        Anaglyph image with matches
    """
    # Ensure both images are the same size
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    w = max(w1, w2)
    
    # Center images
    img1_centered, _, _ = center_image(img1, w, h)
    img2_centered, _, _ = center_image(img2, w, h)
    
    # Ensure color images
    if len(img1_centered.shape) == 2:
        img1_centered = cv2.cvtColor(img1_centered, cv2.COLOR_GRAY2BGR)
    if len(img2_centered.shape) == 2:
        img2_centered = cv2.cvtColor(img2_centered, cv2.COLOR_GRAY2BGR)
    
    # Create anaglyph (Red from left, Green+Blue from right)
    b1, g1, r1 = cv2.split(img1_centered)
    b2, g2, r2 = cv2.split(img2_centered)
    
    anaglyph = cv2.merge([b2, g2, r1])
    
    # Draw matches
    for m in matches:
        kp1 = keypoints1[m.query_idx]
        kp2 = keypoints2[m.train_idx]
        
        pt1 = (int(kp1.x), int(kp1.y))
        pt2 = (int(kp2.x), int(kp2.y))
        
        cv2.line(anaglyph, pt1, pt2, line_color, line_thickness, cv2.LINE_AA)
        cv2.circle(anaglyph, pt1, 3, kp1_color, 1)
        cv2.circle(anaglyph, pt2, 3, kp2_color, 1)
    
    return anaglyph

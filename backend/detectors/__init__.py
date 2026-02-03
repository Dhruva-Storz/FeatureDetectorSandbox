from .base import BaseDetector
from .opencv_detectors import ORBDetector, AKAZEDetector, BRISKDetector, SIFTDetector, SURFDetector

AVAILABLE_DETECTORS = {
    "ORB": ORBDetector,
    "AKAZE": AKAZEDetector,
    "BRISK": BRISKDetector,
    "SIFT": SIFTDetector,
    "SURF": SURFDetector,
}

def get_detector(name: str, **kwargs) -> BaseDetector:
    """Factory function to get a detector by name."""
    if name not in AVAILABLE_DETECTORS:
        raise ValueError(f"Unknown detector: {name}. Available: {list(AVAILABLE_DETECTORS.keys())}")
    return AVAILABLE_DETECTORS[name](**kwargs)

def list_detectors() -> list[dict]:
    """List all available detectors with their info."""
    detectors = []
    for name, cls in AVAILABLE_DETECTORS.items():
        try:
            instance = cls()
            detectors.append({
                "name": name,
                "description": instance.description,
                "available": True
            })
        except Exception as e:
            detectors.append({
                "name": name,
                "description": str(e),
                "available": False
            })
    return detectors

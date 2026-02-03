"""Dense/End-to-End matchers that combine detection and matching in one step."""
from .base import BaseDenseMatcher, DenseMatchResult

# Import implementations - these may fail if dependencies aren't installed
AVAILABLE_DENSE_MATCHERS = {}

try:
    from .lightglue_matcher import (
        SuperPointLightGlueMatcher,
        DISKLightGlueMatcher,
        LightGlueMatcher,  # Alias for backward compat
    )
    AVAILABLE_DENSE_MATCHERS["SuperPoint+LightGlue"] = SuperPointLightGlueMatcher
    AVAILABLE_DENSE_MATCHERS["DISK+LightGlue"] = DISKLightGlueMatcher
    # Keep backward compat alias
    AVAILABLE_DENSE_MATCHERS["LightGlue"] = SuperPointLightGlueMatcher
except ImportError:
    pass

try:
    from .efficientloftr_matcher import EfficientLoFTRMatcher
    AVAILABLE_DENSE_MATCHERS["EfficientLoFTR"] = EfficientLoFTRMatcher
except ImportError:
    pass


def get_dense_matcher(name: str, **kwargs) -> BaseDenseMatcher:
    """Factory function to get a dense matcher by name."""
    if name not in AVAILABLE_DENSE_MATCHERS:
        available = list(AVAILABLE_DENSE_MATCHERS.keys())
        raise ValueError(f"Unknown dense matcher: {name}. Available: {available}")
    return AVAILABLE_DENSE_MATCHERS[name](**kwargs)


def list_dense_matchers() -> list[dict]:
    """List all available dense matchers with their info."""
    matchers = []

    # All known dense matchers (in display order)
    all_matchers = [
        ("SuperPoint+LightGlue", "SuperPoint + LightGlue - Fast learned feature matcher"),
        ("DISK+LightGlue", "DISK + LightGlue - Discrete keypoint feature matcher"),
        ("EfficientLoFTR", "EfficientLoFTR - Semi-dense detector-free matcher"),
    ]

    for name, description in all_matchers:
        if name in AVAILABLE_DENSE_MATCHERS:
            try:
                instance = AVAILABLE_DENSE_MATCHERS[name]()
                matchers.append({
                    "name": name,
                    "description": instance.description,
                    "available": True
                })
            except Exception as e:
                matchers.append({
                    "name": name,
                    "description": f"{description} (Error: {str(e)})",
                    "available": False
                })
        else:
            matchers.append({
                "name": name,
                "description": f"{description} (Not installed - run: pip install torch transformers)",
                "available": False
            })

    return matchers

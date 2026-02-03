from .base import BaseMatcher
from .opencv_matchers import BFMatcher

AVAILABLE_MATCHERS = {
    "BF": BFMatcher,
}

def get_matcher(name: str, **kwargs) -> BaseMatcher:
    """Factory function to get a matcher by name."""
    if name not in AVAILABLE_MATCHERS:
        raise ValueError(f"Unknown matcher: {name}. Available: {list(AVAILABLE_MATCHERS.keys())}")
    return AVAILABLE_MATCHERS[name](**kwargs)

def list_matchers() -> list[dict]:
    """List all available matchers with their info."""
    matchers = []
    for name, cls in AVAILABLE_MATCHERS.items():
        try:
            instance = cls()
            matchers.append({
                "name": name,
                "description": instance.description,
                "available": True
            })
        except Exception as e:
            matchers.append({
                "name": name,
                "description": str(e),
                "available": False
            })
    return matchers

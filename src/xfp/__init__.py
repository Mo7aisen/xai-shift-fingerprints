"""Attribution fingerprint toolkit package."""

from importlib.metadata import version, PackageNotFoundError


def get_version() -> str:
    """Return package version, falling back to development placeholder."""
    try:
        return version("xai-shift-fingerprints")
    except PackageNotFoundError:  # pragma: no cover - during local dev w/out install
        return "0.1.0-dev"


__all__ = ["get_version"]

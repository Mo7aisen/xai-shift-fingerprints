"""Path setup helper for scripts.

This module provides a clean way to ensure the xfp package is importable.
It tries the following in order:
1. Import xfp normally (works if installed via `pip install -e .` or `poetry install`)
2. Add the src directory to sys.path (development fallback)

Usage in scripts:
    import _path_setup  # noqa: F401 - ensures xfp is importable
    from xfp.config import load_paths_config  # Now works
"""

from __future__ import annotations

import sys
from pathlib import Path


def _setup_xfp_import() -> None:
    """Ensure xfp package is importable."""
    try:
        import xfp  # noqa: F401
        return  # Already importable, nothing to do
    except ImportError:
        pass

    # Fallback: add src directory to path
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"

    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # Verify import now works
    try:
        import xfp  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"Failed to import xfp package. Please either:\n"
            f"  1. Install the package: pip install -e {project_root}\n"
            f"  2. Or ensure {src_dir} exists and contains the xfp package\n"
            f"Original error: {e}"
        ) from e


# Run setup on import
_setup_xfp_import()

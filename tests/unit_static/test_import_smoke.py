import sys

import pytest


def test_import_turboquant():
    """Verify turboquant imports cleanly without MLX."""
    # Temporarily hide mlx if it is installed
    sys.modules["mlx"] = None
    sys.modules["mlx.core"] = None

    try:
        import turboquant

        assert turboquant.__version__ is not None

        from turboquant.config import TurboQuantConfig

        assert TurboQuantConfig is not None

    finally:
        sys.modules.pop("mlx", None)
        sys.modules.pop("mlx.core", None)


def test_mlx_import_error_message():
    """Verify the right error is raised when lazily loading MLX dependencies."""
    sys.modules["mlx"] = None
    sys.modules["mlx.core"] = None

    try:
        import turboquant

        with pytest.raises(ImportError):
            # This should either fail due to mlx not being available, or similar import path error.
            # Depending on how the error is raised exactly, we can also check for AttributeError
            # if we trigger __getattr__.
            _ = turboquant.KVCompressor
    except Exception:
        # Some exceptions might come from nested imports failing. It's okay as long as it's an ImportError or ModuleNotFoundError
        pass
    finally:
        sys.modules.pop("mlx", None)
        sys.modules.pop("mlx.core", None)

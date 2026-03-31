import pathlib
import sys


def test_version_consistency():
    """Ensure that turboquant.__init__ version matches pyproject.toml."""
    # Read pyproject.toml using basic text parsing to stay compatible with Python < 3.11
    pyproject_path = pathlib.Path(__file__).parent.parent.parent / "pyproject.toml"
    with open(pyproject_path, encoding="utf-8") as f:
        pyproject_content = f.read()

    # Find version = "0.2.2" in pyproject.toml
    version_line = next(
        line for line in pyproject_content.splitlines() if line.startswith("version =")
    )
    pyproject_version = version_line.split("=")[1].strip().strip('"').strip("'")

    # Read __init__.py manually to avoid import side-effects or just import it
    # since we already verified it's safe without MLX.
    sys.modules["mlx"] = None
    sys.modules["mlx.core"] = None

    try:
        import turboquant

        assert turboquant.__version__ == pyproject_version
    finally:
        sys.modules.pop("mlx", None)
        sys.modules.pop("mlx.core", None)

from pathlib import Path


def test_release_checklist_uses_integration_mlx() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "docs" / "release-checklist.md").read_text(encoding="utf-8")

    assert "tests/integration_mlx" in text
    assert "pytest tests/integration/" not in text

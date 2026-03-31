import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "typecheck", "tests_static"]
nox.options.reuse_existing_virtualenvs = True

PYTHON_VERSIONS = ["3.9", "3.10", "3.11"]


@nox.session(python=PYTHON_VERSIONS)
def tests_static(session: nox.Session) -> None:
    """Run generic static tests without MLX dependency."""
    session.install(".[test]")
    session.run("pytest", "tests/unit_static/", *session.posargs)


@nox.session(python=PYTHON_VERSIONS)
def tests_mlx(session: nox.Session) -> None:
    """Run the Apple-Silicon MLX-supported test surface."""
    session.install(".[test,apple]")
    session.run(
        "pytest",
        "--cov=turboquant",
        "--cov-report=term-missing",
        "tests/integration_mlx/",
        *session.posargs,
    )


@nox.session(python="3.11")
def lint(session: nox.Session) -> None:
    """Run linting using ruff."""
    session.install("ruff")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session(python="3.11")
def typecheck(session: nox.Session) -> None:
    """Run type checking with mypy."""
    session.install("mypy", ".")
    session.run("mypy", "turboquant/", "mlx_lm/")

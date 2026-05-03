"""Allow ``python -m derail.cli`` invocation in addition to the
installed ``derail`` console script."""

from derail.cli import app

if __name__ == "__main__":
    app()

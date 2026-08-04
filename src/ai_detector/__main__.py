# CLI entry point for running the package with python -m ai_detector.
from .cli import main


# Start the command-line app when this module is executed directly.
if __name__ == "__main__":
    raise SystemExit(main())

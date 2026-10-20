"""``python -m devagent`` / ``devagent`` entry point."""
from __future__ import annotations

import sys

from .cli.repl import main

if __name__ == "__main__":
    sys.exit(main())

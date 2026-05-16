from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_import_path() -> None:
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> int:
    _bootstrap_import_path()
    from repair_cost_cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())

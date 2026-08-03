#!/usr/bin/env python3
"""Launch the SplatFormer evaluation image viewer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True, help="Evaluation root containing numeric iteration folders")
    parser.add_argument("--reference-iteration", default="00000000", help="Iteration containing input/GT compare strips")
    parser.add_argument("--host", default="127.0.0.1", help="Server bind address")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("FastAPI viewer dependencies are missing. Install eval_viewer/requirements.txt") from exc

    from eval_viewer.server import create_app

    app = create_app(args.eval_dir, reference_iteration=args.reference_iteration)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

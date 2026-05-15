from __future__ import annotations

import argparse
import json
from pathlib import Path

from aruco_measurement_app.main_window import launch_app
from aruco_measurement_app.processing import run_self_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure crack growth from specimen images using ArUco calibration.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a headless smoke test against the bundled sample images and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        root_dir = Path(__file__).resolve().parent.parent
        result = run_self_test(root_dir)
        print(json.dumps(result, indent=2))
        return 0

    return launch_app()


if __name__ == "__main__":
    raise SystemExit(main())


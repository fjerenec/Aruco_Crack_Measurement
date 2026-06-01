from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path


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


def write_startup_crash_report(exc: BaseException) -> Path | None:
    try:
        local_app_data = Path(
            os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        )
        report_dir = local_app_data / "ArucoCrackMeasurement" / "crash_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / (
            f"startup_crash_{datetime.now():%Y%m%d_%H%M%S}.txt"
        )
        traceback_text = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        report_body = "\n".join(
            [
                "Aruco Crack Measurement startup failure",
                f"timestamp: {datetime.now().isoformat()}",
                f"python_executable: {sys.executable}",
                f"frozen: {getattr(sys, 'frozen', False)}",
                f"cwd: {Path.cwd()}",
                "",
                traceback_text,
            ]
        )
        report_path.write_text(report_body, encoding="utf-8")
        return report_path
    except Exception:
        return None


def show_startup_error_dialog(report_path: Path | None) -> None:
    if sys.platform != "win32":
        return

    message_lines = [
        "Aruco Crack Measurement could not start.",
        "",
    ]
    if report_path is not None:
        message_lines.extend(
            [
                "A crash report was written to:",
                str(report_path),
                "",
            ]
        )
    message_lines.append("Please keep this file and share it for troubleshooting.")

    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(
            None,
            "\n".join(message_lines),
            "Aruco Crack Measurement",
            0x10,
        )
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    try:
        from aruco_measurement_app.main_window import launch_app
        from aruco_measurement_app.processing import run_self_test

        if args.self_test:
            root_dir = Path(__file__).resolve().parent.parent
            result = run_self_test(root_dir)
            print(json.dumps(result, indent=2))
            return 0

        return launch_app()
    except Exception as exc:
        report_path = write_startup_crash_report(exc)
        traceback.print_exc()
        show_startup_error_dialog(report_path)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

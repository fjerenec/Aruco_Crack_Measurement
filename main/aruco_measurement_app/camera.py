from __future__ import annotations

import copy
import time
from typing import Any

import cv2
from PySide6.QtCore import QObject, QThread, Signal, Slot

from .models import AppSettings, LiveFrameResult
from .processing import ProcessingError, detect_and_rectify


class CameraWorker(QObject):
    frame_ready = Signal(object)
    finished = Signal()

    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self._settings = copy.deepcopy(settings)
        self._running = True

    def update_settings(self, settings: AppSettings) -> None:
        self._settings = copy.deepcopy(settings)

    def stop(self) -> None:
        self._running = False

    @Slot()
    def run(self) -> None:
        capture = None
        try:
            capture = cv2.VideoCapture(self._parse_camera_source(self._settings.live.camera_source))
            if not capture.isOpened():
                self.frame_ready.emit(
                    LiveFrameResult(
                        original_bgr=None,
                        rectification=None,
                        error=(
                            "Unable to open the configured camera source. "
                            "Check the camera index or source path in the settings panel."
                        ),
                    )
                )
                return

            while self._running:
                loop_started = time.perf_counter()
                current_settings = copy.deepcopy(self._settings)

                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    self.frame_ready.emit(
                        LiveFrameResult(
                            original_bgr=None,
                            rectification=None,
                            error="The camera returned an empty frame.",
                        )
                    )
                    break

                try:
                    rectification = detect_and_rectify(
                        frame_bgr,
                        current_settings.detector,
                        current_settings.calibration,
                    )
                    frame_result = LiveFrameResult(
                        original_bgr=frame_bgr,
                        rectification=rectification,
                        error=None,
                    )
                except ProcessingError as exc:
                    frame_result = LiveFrameResult(
                        original_bgr=frame_bgr,
                        rectification=None,
                        error=str(exc),
                    )

                self.frame_ready.emit(frame_result)
                target_frame_time = 1.0 / max(1, current_settings.live.target_fps)
                remaining = target_frame_time - (time.perf_counter() - loop_started)
                if remaining > 0:
                    QThread.msleep(int(remaining * 1000))
        finally:
            if capture is not None:
                capture.release()
            self.finished.emit()

    @staticmethod
    def _parse_camera_source(source: str) -> Any:
        source = source.strip()
        if source.isdigit():
            return int(source)
        return source

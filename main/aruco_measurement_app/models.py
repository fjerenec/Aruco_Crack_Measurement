from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CalibrationSettings:
    marker_width_mm: float = 9.915
    marker_height_mm: float = 9.987
    top_pin_dx_mm: float = 0.0
    top_pin_dy_mm: float = 0.0
    bottom_pin_dx_mm: float = 0.0
    bottom_pin_dy_mm: float = 0.0
    rectified_scale_px_per_mm: float = 25.0


@dataclass
class DetectorSettings:
    dictionary_name: str = "DICT_4X4_100"
    adaptive_thresh_win_size_min: int = 15
    adaptive_thresh_win_size_max: int = 101
    adaptive_thresh_win_size_step: int = 35
    min_marker_perimeter_rate: float = 0.1
    max_marker_perimeter_rate: float = 100.0
    polygonal_approx_accuracy_rate: float = 0.01
    min_corner_distance_rate: float = 0.01
    corner_refinement: str = "SUBPIX"


@dataclass
class LiveSettings:
    camera_source: str = "0"
    target_fps: int = 15


@dataclass
class AppSettings:
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    detector: DetectorSettings = field(default_factory=DetectorSettings)
    live: LiveSettings = field(default_factory=LiveSettings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration": vars(self.calibration).copy(),
            "detector": vars(self.detector).copy(),
            "live": vars(self.live).copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppSettings":
        calibration = CalibrationSettings(**data.get("calibration", {}))
        detector = DetectorSettings(**data.get("detector", {}))
        live = LiveSettings(**data.get("live", {}))
        return cls(calibration=calibration, detector=detector, live=live)


@dataclass
class MarkerDetection:
    marker_id: int
    corners_px: np.ndarray
    detected_marker_count: int
    rejected_candidate_count: int
    corner_source: str = "aruco-direct"


@dataclass
class RectificationResult:
    image_bgr: np.ndarray
    rectified_bgr: np.ndarray
    homography: np.ndarray
    inverse_homography: np.ndarray
    marker_corners_image_px: np.ndarray
    marker_corners_rectified_px: np.ndarray
    marker_id: int
    detected_marker_count: int
    rejected_candidate_count: int
    corner_source: str = "aruco-direct"


@dataclass
class MeasurementResult:
    crack_tip_mm: np.ndarray
    crack_tip_rectified_px: np.ndarray
    top_pin_mm: np.ndarray
    bottom_pin_mm: np.ndarray
    top_pin_rectified_px: np.ndarray
    bottom_pin_rectified_px: np.ndarray
    load_line_projection_mm: np.ndarray
    load_line_projection_rectified_px: np.ndarray
    signed_length_mm: float
    absolute_length_mm: float


@dataclass
class MeasurementRecord:
    image_path: Path
    cycle_count: int = 0
    saved_tip_mm: np.ndarray | None = None
    saved_signed_length_mm: float | None = None
    saved_absolute_length_mm: float | None = None
    marker_id: int | None = None
    status: str = "Pending"


@dataclass
class PlotDataset:
    name: str
    points: list[tuple[float, float]]
    source_path: Path | None = None


@dataclass
class LiveFrameResult:
    original_bgr: np.ndarray | None
    rectification: RectificationResult | None
    error: str | None = None

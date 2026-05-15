from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import (
    AppSettings,
    CalibrationSettings,
    DetectorSettings,
    MarkerDetection,
    MeasurementResult,
    RectificationResult,
)

ARUCO_DICTIONARIES: dict[str, int] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

CORNER_REFINEMENT_METHODS: dict[str, int] = {
    "NONE": cv2.aruco.CORNER_REFINE_NONE,
    "SUBPIX": cv2.aruco.CORNER_REFINE_SUBPIX,
    "CONTOUR": cv2.aruco.CORNER_REFINE_CONTOUR,
    "APRILTAG": cv2.aruco.CORNER_REFINE_APRILTAG,
}

IMAGE_FILE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


class ProcessingError(RuntimeError):
    """Raised when the image-processing pipeline cannot complete."""


def list_image_files(folder_path: Path) -> list[Path]:
    return sorted(
        path
        for path in folder_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_FILE_EXTENSIONS
    )


def load_color_image(image_path: Path) -> np.ndarray:
    image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ProcessingError(f"Unable to read image: {image_path}")
    return image


def bgr_to_rgb(image_bgr: np.ndarray | None) -> np.ndarray | None:
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def build_detector(detector_settings: DetectorSettings) -> cv2.aruco.ArucoDetector:
    dictionary_key = detector_settings.dictionary_name
    if dictionary_key not in ARUCO_DICTIONARIES:
        raise ProcessingError(f"Unsupported ArUco dictionary: {dictionary_key}")

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary_key])
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = detector_settings.adaptive_thresh_win_size_min
    params.adaptiveThreshWinSizeMax = detector_settings.adaptive_thresh_win_size_max
    params.adaptiveThreshWinSizeStep = detector_settings.adaptive_thresh_win_size_step
    params.minMarkerPerimeterRate = detector_settings.min_marker_perimeter_rate
    params.maxMarkerPerimeterRate = detector_settings.max_marker_perimeter_rate
    params.polygonalApproxAccuracyRate = detector_settings.polygonal_approx_accuracy_rate
    params.minCornerDistanceRate = detector_settings.min_corner_distance_rate
    params.cornerRefinementMethod = CORNER_REFINEMENT_METHODS[
        detector_settings.corner_refinement
    ]
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_primary_marker(
    image_bgr: np.ndarray,
    detector_settings: DetectorSettings,
) -> MarkerDetection:
    detector = build_detector(detector_settings)
    grayscale_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(grayscale_image)

    if ids is None or len(corners) == 0:
        raise ProcessingError("No ArUco marker detected in the current image.")

    areas = [
        cv2.contourArea(marker_corners.reshape(-1, 1, 2).astype(np.float32))
        for marker_corners in corners
    ]
    primary_index = int(np.argmax(areas))
    primary_corners = corners[primary_index].reshape(4, 2).astype(np.float32)
    marker_id = int(ids[primary_index][0])

    return MarkerDetection(
        marker_id=marker_id,
        corners_px=primary_corners,
        detected_marker_count=len(corners),
        rejected_candidate_count=len(rejected),
        corner_source="aruco-direct",
    )


def _transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    transformed = cv2.perspectiveTransform(
        points.reshape(-1, 1, 2).astype(np.float32),
        homography,
    )
    return transformed.reshape(-1, 2)


def _subpixel_extremum_position(signal: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(signal) - 1:
        return float(index)

    y_prev = float(signal[index - 1])
    y_curr = float(signal[index])
    y_next = float(signal[index + 1])
    denominator = y_prev - (2.0 * y_curr) + y_next
    if abs(denominator) < 1e-9:
        return float(index)

    offset = 0.5 * (y_prev - y_next) / denominator
    return float(index + np.clip(offset, -1.0, 1.0))


def _fit_homogeneous_line(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        raise ProcessingError("Not enough edge points were found to fit a marker edge.")

    fit = cv2.fitLine(points.reshape(-1, 1, 2).astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = [float(value) for value in fit.reshape(-1)]
    line = np.array([vy, -vx, (vx * y0) - (vy * x0)], dtype=np.float64)
    line /= np.linalg.norm(line[:2])

    distances = np.abs((points @ line[:2]) + line[2])
    distance_threshold = max(1.0, float(np.median(distances) * 2.5))
    inlier_points = points[distances <= distance_threshold]
    if len(inlier_points) >= max(6, len(points) // 2) and len(inlier_points) < len(points):
        fit = cv2.fitLine(
            inlier_points.reshape(-1, 1, 2).astype(np.float32),
            cv2.DIST_L2,
            0,
            0.01,
            0.01,
        )
        vx, vy, x0, y0 = [float(value) for value in fit.reshape(-1)]
        line = np.array([vy, -vx, (vx * y0) - (vy * x0)], dtype=np.float64)
        line /= np.linalg.norm(line[:2])

    return line


def _intersect_lines(first_line: np.ndarray, second_line: np.ndarray) -> np.ndarray:
    homogeneous_point = np.cross(first_line, second_line)
    if abs(homogeneous_point[2]) < 1e-9:
        raise ProcessingError("The refined marker edge lines were nearly parallel.")
    return (homogeneous_point[:2] / homogeneous_point[2]).astype(np.float32)


def _sample_horizontal_edge_points(
    warped_gray: np.ndarray,
    x_positions: np.ndarray,
    y_start: int,
    y_end: int,
    sample_band_radius: int,
    prefer_positive_gradient: bool,
) -> np.ndarray:
    points: list[list[float]] = []
    image_height, image_width = warped_gray.shape[:2]

    y_start = max(0, y_start)
    y_end = min(image_height, y_end)
    if y_end - y_start < 5:
        raise ProcessingError("Horizontal edge search window was too small.")

    for x_position in x_positions:
        x_index = int(round(float(x_position)))
        x0 = max(0, x_index - sample_band_radius)
        x1 = min(image_width, x_index + sample_band_radius + 1)
        column_band = warped_gray[y_start:y_end, x0:x1]
        if column_band.size == 0:
            continue
        profile = column_band.mean(axis=1).astype(np.float64)
        gradient = np.gradient(profile)
        edge_index = int(np.argmax(gradient) if prefer_positive_gradient else np.argmin(gradient))
        edge_position = y_start + _subpixel_extremum_position(gradient, edge_index)
        points.append([float(x_position), float(edge_position)])

    if len(points) < 2:
        raise ProcessingError("Failed to recover enough horizontal marker edge samples.")
    return np.asarray(points, dtype=np.float32)


def _sample_vertical_edge_points(
    warped_gray: np.ndarray,
    y_positions: np.ndarray,
    x_start: int,
    x_end: int,
    sample_band_radius: int,
    prefer_positive_gradient: bool,
) -> np.ndarray:
    points: list[list[float]] = []
    image_height, image_width = warped_gray.shape[:2]

    x_start = max(0, x_start)
    x_end = min(image_width, x_end)
    if x_end - x_start < 5:
        raise ProcessingError("Vertical edge search window was too small.")

    for y_position in y_positions:
        y_index = int(round(float(y_position)))
        y0 = max(0, y_index - sample_band_radius)
        y1 = min(image_height, y_index + sample_band_radius + 1)
        row_band = warped_gray[y0:y1, x_start:x_end]
        if row_band.size == 0:
            continue
        profile = row_band.mean(axis=0).astype(np.float64)
        gradient = np.gradient(profile)
        edge_index = int(np.argmax(gradient) if prefer_positive_gradient else np.argmin(gradient))
        edge_position = x_start + _subpixel_extremum_position(gradient, edge_index)
        points.append([float(edge_position), float(y_position)])

    if len(points) < 2:
        raise ProcessingError("Failed to recover enough vertical marker edge samples.")
    return np.asarray(points, dtype=np.float32)


def refine_marker_detection(
    image_bgr: np.ndarray,
    detection: MarkerDetection,
    calibration_settings: CalibrationSettings,
) -> MarkerDetection:
    if calibration_settings.marker_width_mm <= 0 or calibration_settings.marker_height_mm <= 0:
        return detection

    grayscale_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    canonical_marker_width = 480
    canonical_marker_height = max(
        320,
        int(
            round(
                canonical_marker_width
                * calibration_settings.marker_height_mm
                / calibration_settings.marker_width_mm
            )
        ),
    )
    padding = max(80, int(round(0.25 * max(canonical_marker_width, canonical_marker_height))))

    destination_points = np.array(
        [
            [padding, padding],
            [padding + canonical_marker_width, padding],
            [padding + canonical_marker_width, padding + canonical_marker_height],
            [padding, padding + canonical_marker_height],
        ],
        dtype=np.float32,
    )

    initial_homography = cv2.getPerspectiveTransform(
        detection.corners_px.astype(np.float32),
        destination_points,
    )
    warped_width = int((2 * padding) + canonical_marker_width)
    warped_height = int((2 * padding) + canonical_marker_height)
    warped_gray = cv2.warpPerspective(
        grayscale_image,
        initial_homography,
        (warped_width, warped_height),
        flags=cv2.INTER_CUBIC,
    )
    warped_gray = cv2.GaussianBlur(warped_gray, (5, 5), 0)

    search_margin = min(
        padding - 8,
        max(20, int(round(0.14 * min(canonical_marker_width, canonical_marker_height)))),
    )
    horizontal_trim = max(24, int(round(0.12 * canonical_marker_width)))
    vertical_trim = max(24, int(round(0.12 * canonical_marker_height)))
    horizontal_sample_positions = np.linspace(
        padding + horizontal_trim,
        padding + canonical_marker_width - horizontal_trim,
        40,
    )
    vertical_sample_positions = np.linspace(
        padding + vertical_trim,
        padding + canonical_marker_height - vertical_trim,
        40,
    )
    sample_band_radius = max(
        2,
        int(round(0.0125 * min(canonical_marker_width, canonical_marker_height))),
    )

    try:
        top_points = _sample_horizontal_edge_points(
            warped_gray,
            horizontal_sample_positions,
            padding - search_margin,
            padding + search_margin,
            sample_band_radius,
            prefer_positive_gradient=False,
        )
        bottom_points = _sample_horizontal_edge_points(
            warped_gray,
            horizontal_sample_positions,
            padding + canonical_marker_height - search_margin,
            padding + canonical_marker_height + search_margin,
            sample_band_radius,
            prefer_positive_gradient=True,
        )
        left_points = _sample_vertical_edge_points(
            warped_gray,
            vertical_sample_positions,
            padding - search_margin,
            padding + search_margin,
            sample_band_radius,
            prefer_positive_gradient=False,
        )
        right_points = _sample_vertical_edge_points(
            warped_gray,
            vertical_sample_positions,
            padding + canonical_marker_width - search_margin,
            padding + canonical_marker_width + search_margin,
            sample_band_radius,
            prefer_positive_gradient=True,
        )

        top_line = _fit_homogeneous_line(top_points)
        bottom_line = _fit_homogeneous_line(bottom_points)
        left_line = _fit_homogeneous_line(left_points)
        right_line = _fit_homogeneous_line(right_points)

        refined_corners_canonical = np.array(
            [
                _intersect_lines(top_line, left_line),
                _intersect_lines(top_line, right_line),
                _intersect_lines(bottom_line, right_line),
                _intersect_lines(bottom_line, left_line),
            ],
            dtype=np.float32,
        )

        refined_corners_original = _transform_points(
            refined_corners_canonical,
            np.linalg.inv(initial_homography),
        )
    except Exception as exc:
        raise ProcessingError("Edge-based marker corner refinement failed.") from exc

    return MarkerDetection(
        marker_id=detection.marker_id,
        corners_px=refined_corners_original.astype(np.float32),
        detected_marker_count=detection.detected_marker_count,
        rejected_candidate_count=detection.rejected_candidate_count,
        corner_source="aruco-edge-refined",
    )


def rectify_image(
    image_bgr: np.ndarray,
    detection: MarkerDetection,
    calibration_settings: CalibrationSettings,
) -> RectificationResult:
    if calibration_settings.marker_width_mm <= 0 or calibration_settings.marker_height_mm <= 0:
        raise ProcessingError("Marker width and height must both be greater than zero.")
    if calibration_settings.rectified_scale_px_per_mm <= 0:
        raise ProcessingError("Rectified scale must be greater than zero.")

    source_points = detection.corners_px.astype(np.float32)
    marker_width_px = (
        calibration_settings.marker_width_mm * calibration_settings.rectified_scale_px_per_mm
    )
    marker_height_px = (
        calibration_settings.marker_height_mm * calibration_settings.rectified_scale_px_per_mm
    )
    destination_points = np.array(
        [
            [0.0, 0.0],
            [marker_width_px, 0.0],
            [marker_width_px, marker_height_px],
            [0.0, marker_height_px],
        ],
        dtype=np.float32,
    )

    homography, _ = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5.0)
    if homography is None:
        raise ProcessingError("Failed to compute the ArUco homography matrix.")

    image_height, image_width = image_bgr.shape[:2]
    image_corners = np.array(
        [
            [0.0, 0.0],
            [float(image_width), 0.0],
            [float(image_width), float(image_height)],
            [0.0, float(image_height)],
        ],
        dtype=np.float32,
    )
    warped_image_corners = _transform_points(image_corners, homography)
    min_corner = np.min(warped_image_corners, axis=0)

    translation = -min_corner
    translated_destination_points = destination_points + translation
    corrected_homography, _ = cv2.findHomography(
        source_points,
        translated_destination_points,
        cv2.RANSAC,
        5.0,
    )
    if corrected_homography is None:
        raise ProcessingError("Failed to compute the translation-corrected homography.")

    corrected_image_corners = _transform_points(image_corners, corrected_homography)
    max_corner = np.max(corrected_image_corners, axis=0)
    warped_width = max(1, int(math.ceil(max_corner[0])))
    warped_height = max(1, int(math.ceil(max_corner[1])))

    rectified_image = cv2.warpPerspective(
        image_bgr,
        corrected_homography,
        (warped_width, warped_height),
    )
    if rectified_image is None or rectified_image.size == 0:
        raise ProcessingError("OpenCV returned an empty rectified image.")

    inverse_homography = np.linalg.inv(corrected_homography)
    rectified_marker_corners = _transform_points(source_points, corrected_homography)

    return RectificationResult(
        image_bgr=image_bgr,
        rectified_bgr=rectified_image,
        homography=corrected_homography,
        inverse_homography=inverse_homography,
        marker_corners_image_px=source_points,
        marker_corners_rectified_px=rectified_marker_corners,
        marker_id=detection.marker_id,
        detected_marker_count=detection.detected_marker_count,
        rejected_candidate_count=detection.rejected_candidate_count,
        corner_source=detection.corner_source,
    )


def detect_and_rectify(
    image_bgr: np.ndarray,
    detector_settings: DetectorSettings,
    calibration_settings: CalibrationSettings,
) -> RectificationResult:
    detection = detect_primary_marker(image_bgr, detector_settings)
    try:
        detection = refine_marker_detection(image_bgr, detection, calibration_settings)
    except ProcessingError:
        pass
    return rectify_image(image_bgr, detection, calibration_settings)


def _marker_basis_vectors(
    marker_corners_rectified_px: np.ndarray,
    calibration_settings: CalibrationSettings,
) -> tuple[np.ndarray, np.ndarray]:
    origin = marker_corners_rectified_px[0]
    x_basis_px_per_mm = (
        marker_corners_rectified_px[1] - marker_corners_rectified_px[0]
    ) / calibration_settings.marker_width_mm
    y_basis_px_per_mm = (
        marker_corners_rectified_px[3] - marker_corners_rectified_px[0]
    ) / calibration_settings.marker_height_mm
    basis_matrix = np.column_stack([x_basis_px_per_mm, y_basis_px_per_mm])
    return origin, basis_matrix


def rectified_pixels_to_marker_mm(
    rectified_point_px: np.ndarray,
    marker_corners_rectified_px: np.ndarray,
    calibration_settings: CalibrationSettings,
) -> np.ndarray:
    origin, basis_matrix = _marker_basis_vectors(
        marker_corners_rectified_px,
        calibration_settings,
    )
    relative_px = np.asarray(rectified_point_px, dtype=np.float64) - origin
    try:
        return np.linalg.solve(basis_matrix, relative_px)
    except np.linalg.LinAlgError as exc:
        raise ProcessingError("Failed to convert rectified pixels to marker coordinates.") from exc


def marker_mm_to_rectified_pixels(
    marker_point_mm: np.ndarray,
    marker_corners_rectified_px: np.ndarray,
    calibration_settings: CalibrationSettings,
) -> np.ndarray:
    origin, basis_matrix = _marker_basis_vectors(
        marker_corners_rectified_px,
        calibration_settings,
    )
    marker_point_mm = np.asarray(marker_point_mm, dtype=np.float64)
    return origin + (basis_matrix @ marker_point_mm)


def project_image_point_to_rectified(
    image_point_px: np.ndarray,
    rectification: RectificationResult,
) -> np.ndarray:
    projected = _transform_points(np.asarray([image_point_px], dtype=np.float32), rectification.homography)
    return projected[0]


def project_rectified_point_to_image(
    rectified_point_px: np.ndarray,
    rectification: RectificationResult,
) -> np.ndarray:
    projected = _transform_points(
        np.asarray([rectified_point_px], dtype=np.float32),
        rectification.inverse_homography,
    )
    return projected[0]


def measure_crack_from_marker_point(
    crack_tip_mm: np.ndarray,
    marker_corners_rectified_px: np.ndarray,
    calibration_settings: CalibrationSettings,
) -> MeasurementResult:
    top_pin_mm = np.array(
        [
            calibration_settings.top_pin_dx_mm,
            calibration_settings.top_pin_dy_mm,
        ],
        dtype=np.float64,
    )
    bottom_pin_mm = np.array(
        [
            calibration_settings.bottom_pin_dx_mm,
            calibration_settings.bottom_pin_dy_mm,
        ],
        dtype=np.float64,
    )
    crack_tip_mm = np.asarray(crack_tip_mm, dtype=np.float64)

    load_line_vector = bottom_pin_mm - top_pin_mm
    load_line_norm = np.linalg.norm(load_line_vector)
    if load_line_norm == 0:
        raise ProcessingError("Top and bottom pin coordinates must not be identical.")

    load_line_direction = load_line_vector / load_line_norm
    crack_direction = np.array([-load_line_direction[1], load_line_direction[0]])
    load_line_projection_mm = top_pin_mm + (
        np.dot(crack_tip_mm - top_pin_mm, load_line_direction) * load_line_direction
    )
    signed_length_mm = float(np.dot(crack_tip_mm - top_pin_mm, crack_direction))

    top_pin_rectified_px = marker_mm_to_rectified_pixels(
        top_pin_mm,
        marker_corners_rectified_px,
        calibration_settings,
    )
    bottom_pin_rectified_px = marker_mm_to_rectified_pixels(
        bottom_pin_mm,
        marker_corners_rectified_px,
        calibration_settings,
    )
    crack_tip_rectified_px = marker_mm_to_rectified_pixels(
        crack_tip_mm,
        marker_corners_rectified_px,
        calibration_settings,
    )
    load_line_projection_rectified_px = marker_mm_to_rectified_pixels(
        load_line_projection_mm,
        marker_corners_rectified_px,
        calibration_settings,
    )

    return MeasurementResult(
        crack_tip_mm=crack_tip_mm,
        crack_tip_rectified_px=crack_tip_rectified_px,
        top_pin_mm=top_pin_mm,
        bottom_pin_mm=bottom_pin_mm,
        top_pin_rectified_px=top_pin_rectified_px,
        bottom_pin_rectified_px=bottom_pin_rectified_px,
        load_line_projection_mm=load_line_projection_mm,
        load_line_projection_rectified_px=load_line_projection_rectified_px,
        signed_length_mm=signed_length_mm,
        absolute_length_mm=abs(signed_length_mm),
    )


def measure_crack_from_rectified_point(
    rectified_point_px: np.ndarray,
    marker_corners_rectified_px: np.ndarray,
    calibration_settings: CalibrationSettings,
) -> MeasurementResult:
    crack_tip_mm = rectified_pixels_to_marker_mm(
        rectified_point_px,
        marker_corners_rectified_px,
        calibration_settings,
    )
    return measure_crack_from_marker_point(
        crack_tip_mm,
        marker_corners_rectified_px,
        calibration_settings,
    )


def measure_crack_from_original_point(
    image_point_px: np.ndarray,
    rectification: RectificationResult,
    calibration_settings: CalibrationSettings,
) -> MeasurementResult:
    rectified_point_px = project_image_point_to_rectified(image_point_px, rectification)
    return measure_crack_from_rectified_point(
        rectified_point_px,
        rectification.marker_corners_rectified_px,
        calibration_settings,
    )


def run_self_test(project_root: Path) -> dict[str, Any]:
    settings = AppSettings()
    sample_paths = [
        project_root / "main" / "aruca.jpg",
        project_root / "Test pics" / "MPB1_pins_01.jpg",
    ]
    results: list[dict[str, Any]] = []
    for sample_path in sample_paths:
        image_bgr = load_color_image(sample_path)
        rectification = detect_and_rectify(
            image_bgr,
            settings.detector,
            settings.calibration,
        )
        results.append(
            {
                "image": str(sample_path),
                "marker_id": rectification.marker_id,
                "corner_source": rectification.corner_source,
                "detected_marker_count": rectification.detected_marker_count,
                "rejected_candidate_count": rectification.rejected_candidate_count,
                "rectified_shape": list(rectification.rectified_bgr.shape),
            }
        )

    return {
        "status": "ok",
        "sample_count": len(results),
        "results": results,
    }

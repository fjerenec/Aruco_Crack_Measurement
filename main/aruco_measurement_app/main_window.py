from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QStandardPaths, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .camera import CameraWorker
from .models import AppSettings, MeasurementRecord, MeasurementResult, PlotDataset
from .processing import (
    ARUCO_DICTIONARIES,
    CORNER_REFINEMENT_METHODS,
    ProcessingError,
    bgr_to_rgb,
    detect_and_rectify,
    list_image_files,
    load_color_image,
    measure_crack_from_marker_point,
    measure_crack_from_original_point,
    measure_crack_from_rectified_point,
    project_rectified_point_to_image,
)
from .widgets import CanvasOverlay, ImageCanvas, OverlayLine, OverlayPoint, OverlayPolygon

MARKER_OVERLAY_COLOR = (255, 215, 0, 175)
PIN_OVERLAY_COLOR = (60, 210, 120, 180)
CRACK_OVERLAY_COLOR = (255, 70, 70, 190)
SESSION_FILE_NAME = "measurement_session_autosave.json"
CHART_SERIES_COLORS = [
    QColor(220, 68, 55),
    QColor(0, 121, 191),
    QColor(44, 160, 44),
    QColor(255, 127, 14),
    QColor(148, 103, 189),
    QColor(140, 86, 75),
]


class SettingsPanel(QWidget):
    apply_requested = Signal()
    load_profile_requested = Signal()
    save_profile_requested = Signal()
    reset_requested = Signal()
    info_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self.set_settings(AppSettings())

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        root_layout.addWidget(scroll_area)

        content = QWidget()
        scroll_area.setWidget(content)
        layout = QVBoxLayout(content)

        calibration_group = QGroupBox("Calibration")
        calibration_form = QFormLayout(calibration_group)
        self.marker_width_spin = self._make_double_spinbox(0.001, 10000.0, 3)
        self.marker_height_spin = self._make_double_spinbox(0.001, 10000.0, 3)
        self.top_pin_dx_spin = self._make_double_spinbox(-10000.0, 10000.0, 3)
        self.top_pin_dy_spin = self._make_double_spinbox(-10000.0, 10000.0, 3)
        self.bottom_pin_dx_spin = self._make_double_spinbox(-10000.0, 10000.0, 3)
        self.bottom_pin_dy_spin = self._make_double_spinbox(-10000.0, 10000.0, 3)
        self.scale_spin = self._make_double_spinbox(1.0, 500.0, 2)
        calibration_form.addRow("Marker width [mm]", self.marker_width_spin)
        calibration_form.addRow("Marker height [mm]", self.marker_height_spin)
        calibration_form.addRow("Top pin dX [mm]", self.top_pin_dx_spin)
        calibration_form.addRow("Top pin dY [mm]", self.top_pin_dy_spin)
        calibration_form.addRow("Bottom pin dX [mm]", self.bottom_pin_dx_spin)
        calibration_form.addRow("Bottom pin dY [mm]", self.bottom_pin_dy_spin)
        calibration_form.addRow("Rectified scale [px/mm]", self.scale_spin)
        layout.addWidget(calibration_group)

        detector_group = QGroupBox("ArUco Detector")
        detector_form = QFormLayout(detector_group)
        self.dictionary_combo = QComboBox()
        self.dictionary_combo.addItems(sorted(ARUCO_DICTIONARIES.keys()))
        self.thresh_min_spin = self._make_int_spinbox(3, 501)
        self.thresh_max_spin = self._make_int_spinbox(3, 1001)
        self.thresh_step_spin = self._make_int_spinbox(1, 250)
        self.min_perimeter_spin = self._make_double_spinbox(0.001, 1000.0, 4)
        self.max_perimeter_spin = self._make_double_spinbox(0.001, 1000.0, 4)
        self.polygon_accuracy_spin = self._make_double_spinbox(0.0001, 10.0, 4)
        self.corner_distance_spin = self._make_double_spinbox(0.0001, 10.0, 4)
        self.corner_refinement_combo = QComboBox()
        self.corner_refinement_combo.addItems(list(CORNER_REFINEMENT_METHODS.keys()))
        detector_form.addRow("Dictionary", self.dictionary_combo)
        detector_form.addRow("Threshold min", self.thresh_min_spin)
        detector_form.addRow("Threshold max", self.thresh_max_spin)
        detector_form.addRow("Threshold step", self.thresh_step_spin)
        detector_form.addRow("Min perimeter rate", self.min_perimeter_spin)
        detector_form.addRow("Max perimeter rate", self.max_perimeter_spin)
        detector_form.addRow("Polygon accuracy", self.polygon_accuracy_spin)
        detector_form.addRow("Min corner distance", self.corner_distance_spin)
        detector_form.addRow("Corner refinement", self.corner_refinement_combo)
        layout.addWidget(detector_group)

        live_group = QGroupBox("Live View")
        live_form = QFormLayout(live_group)
        self.camera_source_edit = QLineEdit()
        self.target_fps_spin = self._make_int_spinbox(1, 60)
        live_form.addRow("Camera source", self.camera_source_edit)
        live_form.addRow("Target FPS", self.target_fps_spin)
        layout.addWidget(live_group)

        self.note_label = QLabel(
            "Tip: click the rectified image to measure directly in marker coordinates. Hover a setting for quick help."
        )
        self.note_label.setWordWrap(True)
        layout.addWidget(self.note_label)

        button_row = QHBoxLayout()
        self.apply_button = QPushButton("Apply")
        self.load_profile_button = QPushButton("Load Profile")
        self.save_profile_button = QPushButton("Save Profile")
        self.reset_button = QPushButton("Reset")
        self.info_button = QPushButton("Information")
        button_row.addWidget(self.apply_button)
        button_row.addWidget(self.load_profile_button)
        button_row.addWidget(self.save_profile_button)
        button_row.addWidget(self.reset_button)
        button_row.addWidget(self.info_button)
        layout.addLayout(button_row)
        layout.addStretch(1)

        self.apply_button.clicked.connect(self.apply_requested.emit)
        self.load_profile_button.clicked.connect(self.load_profile_requested.emit)
        self.save_profile_button.clicked.connect(self.save_profile_requested.emit)
        self.reset_button.clicked.connect(self.reset_requested.emit)
        self.info_button.clicked.connect(self.info_requested.emit)

        self.marker_width_spin.setToolTip(
            "Measured width of the outer black square of the marker in millimetres."
        )
        self.marker_height_spin.setToolTip(
            "Measured height of the outer black square of the marker in millimetres."
        )
        self.top_pin_dx_spin.setToolTip(
            "Marker-frame X coordinate of the top pin center in millimetres."
        )
        self.top_pin_dy_spin.setToolTip(
            "Marker-frame Y coordinate of the top pin center in millimetres."
        )
        self.bottom_pin_dx_spin.setToolTip(
            "Marker-frame X coordinate of the bottom pin center in millimetres."
        )
        self.bottom_pin_dy_spin.setToolTip(
            "Marker-frame Y coordinate of the bottom pin center in millimetres."
        )
        self.scale_spin.setToolTip(
            "Rectified output resolution in pixels per millimetre. Higher values enlarge the rectified image."
        )

        self.dictionary_combo.setToolTip(
            "ArUco dictionary family. This must match the printed marker exactly."
        )
        self.thresh_min_spin.setToolTip(
            "Smallest adaptive-threshold window size OpenCV tries. Lower it for smaller or fainter markers."
        )
        self.thresh_max_spin.setToolTip(
            "Largest adaptive-threshold window size OpenCV tries. Raise it for uneven lighting or glare."
        )
        self.thresh_step_spin.setToolTip(
            "Step between threshold window sizes. Smaller steps test more cases but run more slowly."
        )
        self.min_perimeter_spin.setToolTip(
            "Minimum marker perimeter relative to the image size. Lower it if the marker appears very small."
        )
        self.max_perimeter_spin.setToolTip(
            "Maximum marker perimeter relative to the image size. Lower it if oversized false detections appear."
        )
        self.polygon_accuracy_spin.setToolTip(
            "Contour simplification tolerance. Lower values preserve distorted corners; higher values smooth noise."
        )
        self.corner_distance_spin.setToolTip(
            "Minimum spacing between neighbouring corners. Lower it for tiny markers, raise it if corners merge."
        )
        self.corner_refinement_combo.setToolTip(
            "How OpenCV refines the coarse ArUco guess before the app performs the edge-based corner refinement."
        )

        self.camera_source_edit.setToolTip(
            "Camera index or video source path for the live view."
        )
        self.target_fps_spin.setToolTip(
            "Requested live processing frame rate."
        )

    @staticmethod
    def _make_double_spinbox(minimum: float, maximum: float, decimals: int) -> QDoubleSpinBox:
        spinbox = QDoubleSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setDecimals(decimals)
        spinbox.setSingleStep(0.1)
        return spinbox

    @staticmethod
    def _make_int_spinbox(minimum: int, maximum: int) -> QSpinBox:
        spinbox = QSpinBox()
        spinbox.setRange(minimum, maximum)
        return spinbox

    def get_settings(self) -> AppSettings:
        return AppSettings.from_dict(
            {
                "calibration": {
                    "marker_width_mm": self.marker_width_spin.value(),
                    "marker_height_mm": self.marker_height_spin.value(),
                    "top_pin_dx_mm": self.top_pin_dx_spin.value(),
                    "top_pin_dy_mm": self.top_pin_dy_spin.value(),
                    "bottom_pin_dx_mm": self.bottom_pin_dx_spin.value(),
                    "bottom_pin_dy_mm": self.bottom_pin_dy_spin.value(),
                    "rectified_scale_px_per_mm": self.scale_spin.value(),
                },
                "detector": {
                    "dictionary_name": self.dictionary_combo.currentText(),
                    "adaptive_thresh_win_size_min": self.thresh_min_spin.value(),
                    "adaptive_thresh_win_size_max": self.thresh_max_spin.value(),
                    "adaptive_thresh_win_size_step": self.thresh_step_spin.value(),
                    "min_marker_perimeter_rate": self.min_perimeter_spin.value(),
                    "max_marker_perimeter_rate": self.max_perimeter_spin.value(),
                    "polygonal_approx_accuracy_rate": self.polygon_accuracy_spin.value(),
                    "min_corner_distance_rate": self.corner_distance_spin.value(),
                    "corner_refinement": self.corner_refinement_combo.currentText(),
                },
                "live": {
                    "camera_source": self.camera_source_edit.text().strip() or "0",
                    "target_fps": self.target_fps_spin.value(),
                },
            }
        )

    def set_settings(self, settings: AppSettings) -> None:
        self.marker_width_spin.setValue(settings.calibration.marker_width_mm)
        self.marker_height_spin.setValue(settings.calibration.marker_height_mm)
        self.top_pin_dx_spin.setValue(settings.calibration.top_pin_dx_mm)
        self.top_pin_dy_spin.setValue(settings.calibration.top_pin_dy_mm)
        self.bottom_pin_dx_spin.setValue(settings.calibration.bottom_pin_dx_mm)
        self.bottom_pin_dy_spin.setValue(settings.calibration.bottom_pin_dy_mm)
        self.scale_spin.setValue(settings.calibration.rectified_scale_px_per_mm)

        self.dictionary_combo.setCurrentText(settings.detector.dictionary_name)
        self.thresh_min_spin.setValue(settings.detector.adaptive_thresh_win_size_min)
        self.thresh_max_spin.setValue(settings.detector.adaptive_thresh_win_size_max)
        self.thresh_step_spin.setValue(settings.detector.adaptive_thresh_win_size_step)
        self.min_perimeter_spin.setValue(settings.detector.min_marker_perimeter_rate)
        self.max_perimeter_spin.setValue(settings.detector.max_marker_perimeter_rate)
        self.polygon_accuracy_spin.setValue(settings.detector.polygonal_approx_accuracy_rate)
        self.corner_distance_spin.setValue(settings.detector.min_corner_distance_rate)
        self.corner_refinement_combo.setCurrentText(settings.detector.corner_refinement)

        self.camera_source_edit.setText(settings.live.camera_source)
        self.target_fps_spin.setValue(settings.live.target_fps)


class InfoRibbon(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.addTab(
            self._make_page(
                """
                <h3>Detector Parameters</h3>
                <table cellspacing="0" cellpadding="4">
                  <tr><td><b>Dictionary</b></td><td>Must match the printed marker family exactly.</td></tr>
                  <tr><td><b>Threshold min / max</b></td><td>Adaptive threshold window range. Raise the max for uneven lighting; lower the min for smaller or faint markers.</td></tr>
                  <tr><td><b>Threshold step</b></td><td>How many window sizes OpenCV tries between the min and max. Larger steps are faster; smaller steps are more exhaustive.</td></tr>
                  <tr><td><b>Min perimeter rate</b></td><td>Smallest allowed marker size relative to the full image. Lower it when the marker appears tiny.</td></tr>
                  <tr><td><b>Max perimeter rate</b></td><td>Largest allowed marker size relative to the full image. Lower it if very large false detections appear.</td></tr>
                  <tr><td><b>Polygon accuracy</b></td><td>Contour simplification tolerance. Lower values help preserve distorted corners; higher values can suppress noisy contours.</td></tr>
                  <tr><td><b>Min corner distance</b></td><td>Minimum spacing between neighboring corners. Lower it for small markers; raise it if corners collapse together.</td></tr>
                  <tr><td><b>Corner refinement</b></td><td>Post-processing for the initial ArUco guess. <b>SUBPIX</b> is usually best for metrology.</td></tr>
                </table>
                <p>The app now uses <b>ArUco for coarse localization</b> and then refines the <b>outer black-square edges</b> before inferring the final corners.</p>
                """
            ),
            "Detector Parameters",
        )
        self.tabs.addTab(
            self._make_page(
                """
                <h3>Accuracy Tips</h3>
                <ul>
                  <li>Measure the <b>black square</b> width and height separately and use those numbers in calibration.</li>
                  <li>Keep the marker flat, clean, and on the <b>same side of the specimen</b> as the crack you are measuring.</li>
                  <li>Pause motion, lock focus, and keep exposure stable. Motion blur and glare bias both the edge fit and the crack-tip click.</li>
                  <li>If the crack-tip click starts dominating the error, zoom into a tighter field of view or use a higher-resolution still image for saved measurements.</li>
                  <li>The displayed red segment is the <b>actual perpendicular crack-length segment</b> from the pin line to the chosen crack tip.</li>
                </ul>
                """
            ),
            "Accuracy Tips",
        )
        self.tabs.addTab(
            self._make_page(
                """
                <h3>Quick Troubleshooting</h3>
                <ul>
                  <li><b>Marker not found:</b> verify the dictionary, reduce glare, lower the minimum perimeter rate, and widen the threshold range.</li>
                  <li><b>Marker outline looks wrong:</b> check whether the white quiet zone is visible and whether saturation is clipping the black/white edge.</li>
                  <li><b>Too many false detections:</b> raise the minimum perimeter rate, raise the minimum corner distance, or narrow the threshold range.</li>
                  <li><b>Measurements drift:</b> verify the entered marker dimensions, keep the same camera setup, and compare the refined edge position to the visible black-square boundary.</li>
                </ul>
                """
            ),
            "Troubleshooting",
        )
        self.tabs.setMaximumHeight(230)
        layout.addWidget(self.tabs)

    @staticmethod
    def _make_page(html: str) -> QTextBrowser:
        browser = QTextBrowser()
        browser.setReadOnly(True)
        browser.setOpenExternalLinks(False)
        browser.setHtml(html)
        return browser


class MeasurementTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        self.load_files_button = QPushButton("Load Files")
        self.load_folder_button = QPushButton("Load Folder")
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.reprocess_button = QPushButton("Reprocess")
        self.save_button = QPushButton("Save Measurement")
        self.export_button = QPushButton("Export TSV")
        for widget in (
            self.load_files_button,
            self.load_folder_button,
            self.prev_button,
            self.next_button,
            self.reprocess_button,
            self.save_button,
            self.export_button,
        ):
            toolbar.addWidget(widget)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.original_canvas = ImageCanvas("Load images, then click the original or rectified view to choose the crack tip.")
        self.rectified_canvas = ImageCanvas("The rectified measurement view will appear here after detection succeeds.")
        splitter.addWidget(self._wrap_canvas("Original Image", self.original_canvas))
        splitter.addWidget(self._wrap_canvas("Rectified Image", self.rectified_canvas))
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, stretch=3)

        summary_layout = QHBoxLayout()
        self.current_file_label = QLabel("Current image: --")
        self.cycle_spin = QSpinBox()
        self.cycle_spin.setRange(0, 2_000_000_000)
        self.absolute_length_label = QLabel("Absolute length: --")
        self.signed_length_label = QLabel("Signed length: --")
        self.tip_position_label = QLabel("Crack tip [mm]: --")
        summary_layout.addWidget(self.current_file_label, stretch=2)
        summary_layout.addWidget(QLabel("Cycles"))
        summary_layout.addWidget(self.cycle_spin)
        summary_layout.addWidget(self.absolute_length_label)
        summary_layout.addWidget(self.signed_length_label)
        summary_layout.addWidget(self.tip_position_label)
        layout.addLayout(summary_layout)

        self.status_label = QLabel("Load images to begin.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            [
                "Image",
                "Cycles",
                "Absolute [mm]",
                "Signed [mm]",
                "Status",
            ]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 5):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, stretch=1)

    @staticmethod
    def _wrap_canvas(title: str, canvas: ImageCanvas) -> QWidget:
        wrapper = QGroupBox(title)
        layout = QVBoxLayout(wrapper)
        layout.addWidget(canvas)
        return wrapper


class LiveViewTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.clear_tip_button = QPushButton("Clear Crack Tip")
        toolbar.addWidget(self.start_button)
        toolbar.addWidget(self.stop_button)
        toolbar.addWidget(self.clear_tip_button)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.original_canvas = ImageCanvas("The live camera stream will appear here.")
        self.rectified_canvas = ImageCanvas("Click the live rectified image to track a crack tip in marker coordinates.")
        splitter.addWidget(MeasurementTab._wrap_canvas("Live Camera", self.original_canvas))
        splitter.addWidget(MeasurementTab._wrap_canvas("Rectified Live View", self.rectified_canvas))
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, stretch=3)

        summary_layout = QHBoxLayout()
        self.absolute_length_label = QLabel("Absolute length: --")
        self.signed_length_label = QLabel("Signed length: --")
        self.tip_position_label = QLabel("Crack tip [mm]: --")
        self.marker_label = QLabel("Marker: --")
        summary_layout.addWidget(self.absolute_length_label)
        summary_layout.addWidget(self.signed_length_label)
        summary_layout.addWidget(self.tip_position_label)
        summary_layout.addWidget(self.marker_label)
        summary_layout.addStretch(1)
        layout.addLayout(summary_layout)

        self.status_label = QLabel("Camera is idle.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)


class CrackGrowthTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        self.import_button = QPushButton("Import CSV/TSV")
        self.clear_button = QPushButton("Clear Imported")
        self.refresh_button = QPushButton("Refresh Graph")
        toolbar.addWidget(self.import_button)
        toolbar.addWidget(self.clear_button)
        toolbar.addWidget(self.refresh_button)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        layout.addWidget(self.chart_view, stretch=1)

        self.status_label = QLabel(
            "The graph shows crack length [mm] versus cycle count from the current image table and any imported datasets."
        )
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Aruco Crack Measurement")
        self.resize(1600, 980)

        self.measurement_records: list[MeasurementRecord] = []
        self.current_measurement_index: int | None = None
        self.current_measurement_image_bgr = None
        self.current_measurement_rectification = None
        self.current_provisional_measurement: MeasurementResult | None = None
        self.latest_live_original_bgr = None
        self.latest_live_rectification = None
        self.live_tip_mm: np.ndarray | None = None
        self.imported_plot_datasets: list[PlotDataset] = []
        self.camera_thread = None
        self.camera_worker = None
        self._updating_measurement_table = False

        self._build_ui()
        self.statusBar().showMessage("Ready.")
        self.restore_autosaved_session_if_available()

    def _build_ui(self) -> None:
        self.settings_panel = SettingsPanel()
        self.settings_panel.apply_requested.connect(self.apply_settings)
        self.settings_panel.load_profile_requested.connect(self.load_profile)
        self.settings_panel.save_profile_requested.connect(self.save_profile)
        self.settings_panel.reset_requested.connect(self.reset_settings)
        self.settings_panel.info_requested.connect(self.show_information_panel)

        settings_dock = QDockWidget("Calibration and Detector Settings", self)
        settings_dock.setWidget(self.settings_panel)
        settings_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, settings_dock)

        self.info_panel = InfoRibbon()
        self.info_dock = QDockWidget("Information", self)
        self.info_dock.setWidget(self.info_panel)
        self.info_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.info_dock)
        self.tabifyDockWidget(settings_dock, self.info_dock)
        self.info_dock.hide()

        view_menu = self.menuBar().addMenu("View")
        settings_action = settings_dock.toggleViewAction()
        settings_action.setText("Settings Panel")
        info_action = self.info_dock.toggleViewAction()
        info_action.setText("Information Panel")
        view_menu.addAction(settings_action)
        view_menu.addAction(info_action)

        self.tabs = QTabWidget()
        self.measurement_tab = MeasurementTab()
        self.live_tab = LiveViewTab()
        self.graph_tab = CrackGrowthTab()
        self.tabs.addTab(self.live_tab, "Live View")
        self.tabs.addTab(self.measurement_tab, "Image Measurement")
        self.tabs.addTab(self.graph_tab, "Crack Growth")
        self.setCentralWidget(self.tabs)

        self.measurement_tab.load_files_button.clicked.connect(self.load_measurement_files)
        self.measurement_tab.load_folder_button.clicked.connect(self.load_measurement_folder)
        self.measurement_tab.prev_button.clicked.connect(self.select_previous_measurement)
        self.measurement_tab.next_button.clicked.connect(self.select_next_measurement)
        self.measurement_tab.reprocess_button.clicked.connect(self.reprocess_current_measurement)
        self.measurement_tab.save_button.clicked.connect(self.save_current_measurement)
        self.measurement_tab.export_button.clicked.connect(self.export_measurements)
        self.measurement_tab.cycle_spin.valueChanged.connect(self.handle_cycle_count_changed)
        self.measurement_tab.table.itemSelectionChanged.connect(self.on_measurement_selection_changed)
        self.measurement_tab.original_canvas.clicked.connect(self.handle_measurement_original_click)
        self.measurement_tab.rectified_canvas.clicked.connect(self.handle_measurement_rectified_click)
        self.graph_tab.import_button.clicked.connect(self.import_graph_datasets)
        self.graph_tab.clear_button.clicked.connect(self.clear_imported_graph_datasets)
        self.graph_tab.refresh_button.clicked.connect(self.refresh_crack_growth_chart)

        self.live_tab.start_button.clicked.connect(self.start_camera)
        self.live_tab.stop_button.clicked.connect(self.stop_camera)
        self.live_tab.clear_tip_button.clicked.connect(self.clear_live_tip)
        self.live_tab.rectified_canvas.clicked.connect(self.handle_live_rectified_click)

        self.refresh_crack_growth_chart()

    def show_information_panel(self) -> None:
        self.info_dock.show()
        self.info_dock.raise_()

    def current_settings(self) -> AppSettings:
        return self.settings_panel.get_settings()

    def session_file_path(self) -> Path:
        app_data_path = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppDataLocation
        )
        base_path = Path(app_data_path) if app_data_path else Path.cwd()
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path / SESSION_FILE_NAME

    @staticmethod
    def _measurement_record_to_dict(record: MeasurementRecord) -> dict[str, object]:
        return {
            "image_path": str(record.image_path),
            "cycle_count": record.cycle_count,
            "saved_tip_mm": record.saved_tip_mm.tolist() if record.saved_tip_mm is not None else None,
            "saved_signed_length_mm": record.saved_signed_length_mm,
            "saved_absolute_length_mm": record.saved_absolute_length_mm,
            "marker_id": record.marker_id,
            "status": record.status,
        }

    @staticmethod
    def _measurement_record_from_dict(data: dict[str, object]) -> MeasurementRecord:
        saved_tip = data.get("saved_tip_mm")
        saved_tip_mm = (
            np.asarray(saved_tip, dtype=np.float64)
            if isinstance(saved_tip, list) and len(saved_tip) == 2
            else None
        )
        return MeasurementRecord(
            image_path=Path(str(data.get("image_path", ""))),
            cycle_count=int(data.get("cycle_count", 0)),
            saved_tip_mm=saved_tip_mm,
            saved_signed_length_mm=(
                float(data["saved_signed_length_mm"])
                if data.get("saved_signed_length_mm") is not None
                else None
            ),
            saved_absolute_length_mm=(
                float(data["saved_absolute_length_mm"])
                if data.get("saved_absolute_length_mm") is not None
                else None
            ),
            marker_id=(
                int(data["marker_id"])
                if data.get("marker_id") is not None and str(data.get("marker_id")) != ""
                else None
            ),
            status=str(data.get("status", "Pending")),
        )

    @staticmethod
    def _plot_dataset_to_dict(dataset: PlotDataset) -> dict[str, object]:
        return {
            "name": dataset.name,
            "source_path": str(dataset.source_path) if dataset.source_path is not None else None,
            "points": [[float(x_value), float(y_value)] for x_value, y_value in dataset.points],
        }

    @staticmethod
    def _plot_dataset_from_dict(data: dict[str, object]) -> PlotDataset:
        raw_points = data.get("points", [])
        points: list[tuple[float, float]] = []
        if isinstance(raw_points, list):
            for point in raw_points:
                if isinstance(point, list) and len(point) == 2:
                    points.append((float(point[0]), float(point[1])))
        source_path_value = data.get("source_path")
        return PlotDataset(
            name=str(data.get("name", "Imported dataset")),
            points=points,
            source_path=Path(str(source_path_value)) if source_path_value else None,
        )

    def build_session_payload(self) -> dict[str, object]:
        return {
            "version": 1,
            "settings": self.current_settings().to_dict(),
            "measurement_records": [
                self._measurement_record_to_dict(record) for record in self.measurement_records
            ],
            "current_measurement_index": self.current_measurement_index,
            "selected_tab_index": self.tabs.currentIndex(),
            "live_tip_mm": self.live_tip_mm.tolist() if self.live_tip_mm is not None else None,
            "imported_plot_datasets": [
                self._plot_dataset_to_dict(dataset) for dataset in self.imported_plot_datasets
            ],
        }

    def autosave_session(self) -> None:
        session_path = self.session_file_path()
        payload = self.build_session_payload()
        temp_path = session_path.with_suffix(session_path.suffix + ".tmp")
        try:
            temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temp_path.replace(session_path)
        except OSError as exc:
            self.statusBar().showMessage(f"Autosave failed: {exc}", 8000)

    def discard_autosaved_session(self) -> None:
        session_path = self.session_file_path()
        try:
            if session_path.exists():
                session_path.unlink()
        except OSError:
            pass

    def restore_autosaved_session_if_available(self) -> None:
        session_path = self.session_file_path()
        if not session_path.exists():
            return

        try:
            payload = json.loads(session_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.warning(
                self,
                "Autosave restore failed",
                f"Could not read the autosaved session file.\n\n{exc}",
            )
            return

        records_payload = payload.get("measurement_records", [])
        imported_payload = payload.get("imported_plot_datasets", [])
        if not isinstance(records_payload, list):
            return
        if not records_payload and not isinstance(imported_payload, list):
            return
        if not records_payload and not imported_payload:
            return

        answer = QMessageBox.question(
            self,
            "Restore autosaved session?",
            (
                "An autosaved measurement session was found.\n\n"
                f"Restore it from:\n{session_path}\n\n"
                "Choose No to start fresh and delete that autosave file."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.discard_autosaved_session()
            return

        try:
            settings_payload = payload.get("settings", {})
            if isinstance(settings_payload, dict):
                self.settings_panel.set_settings(AppSettings.from_dict(settings_payload))

            self.measurement_records = [
                self._measurement_record_from_dict(record_data)
                for record_data in records_payload
                if isinstance(record_data, dict)
            ]
            self.live_tip_mm = (
                np.asarray(payload["live_tip_mm"], dtype=np.float64)
                if isinstance(payload.get("live_tip_mm"), list)
                and len(payload["live_tip_mm"]) == 2
                else None
            )
            self.imported_plot_datasets = [
                self._plot_dataset_from_dict(dataset_data)
                for dataset_data in imported_payload
                if isinstance(dataset_data, dict)
            ]
            self.current_measurement_index = None
            self.refresh_measurement_table()

            if self.measurement_records:
                selected_index = int(payload.get("current_measurement_index", 0) or 0)
                selected_index = min(max(selected_index, 0), len(self.measurement_records) - 1)
                self.select_measurement_row(selected_index)

            selected_tab_index = int(payload.get("selected_tab_index", 0) or 0)
            if 0 <= selected_tab_index < self.tabs.count():
                self.tabs.setCurrentIndex(selected_tab_index)
            self.refresh_crack_growth_chart()
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.warning(
                self,
                "Autosave restore failed",
                f"Could not restore the autosaved session.\n\n{exc}",
            )
            return

        self.statusBar().showMessage(f"Restored autosaved session from {session_path}", 8000)

    def refresh_measurement_row(self, row: int) -> None:
        if not (0 <= row < len(self.measurement_records)):
            return

        record = self.measurement_records[row]
        self._updating_measurement_table = True
        self._set_table_item(row, 0, record.image_path.name)
        self._set_table_item(row, 1, str(record.cycle_count))
        self._set_table_item(row, 2, self._format_float(record.saved_absolute_length_mm))
        self._set_table_item(row, 3, self._format_float(record.saved_signed_length_mm))
        self._set_table_item(row, 4, record.status)
        self._updating_measurement_table = False

    def store_current_measurement_record(self, require_measurement: bool = False) -> bool:
        if self.current_measurement_index is None:
            return False

        record = self.measurement_records[self.current_measurement_index]
        record.cycle_count = self.measurement_tab.cycle_spin.value()

        if self.current_provisional_measurement is None:
            if require_measurement and record.saved_tip_mm is None:
                QMessageBox.information(
                    self,
                    "Nothing to save",
                    "Choose a crack tip on the current image before saving the measurement.",
                )
                return False
            self.refresh_measurement_row(self.current_measurement_index)
            return True

        record.saved_tip_mm = self.current_provisional_measurement.crack_tip_mm.copy()
        record.saved_signed_length_mm = self.current_provisional_measurement.signed_length_mm
        record.saved_absolute_length_mm = self.current_provisional_measurement.absolute_length_mm
        record.marker_id = (
            self.current_measurement_rectification.marker_id
            if self.current_measurement_rectification is not None
            else None
        )
        record.status = "Measured"
        self.refresh_measurement_row(self.current_measurement_index)
        return True

    @staticmethod
    def _normalize_header_name(header: str) -> str:
        return "".join(character for character in header.lower() if character.isalnum())

    @classmethod
    def _pick_row_value(
        cls,
        row: dict[str, str],
        candidate_headers: tuple[str, ...],
    ) -> str | None:
        normalized_row = {
            cls._normalize_header_name(key): value
            for key, value in row.items()
            if key is not None and value is not None
        }
        for header in candidate_headers:
            value = normalized_row.get(cls._normalize_header_name(header))
            if value is not None and str(value).strip() != "":
                return str(value).strip()
        return None

    def current_measurement_plot_dataset(self) -> PlotDataset:
        points = [
            (float(record.cycle_count), float(record.saved_absolute_length_mm))
            for record in self.measurement_records
            if record.saved_absolute_length_mm is not None
        ]
        points.sort(key=lambda point: point[0])
        return PlotDataset(name="Current measurements", points=points)

    def load_plot_dataset_from_file(self, dataset_path: Path) -> PlotDataset:
        try:
            raw_text = dataset_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            raw_text = dataset_path.read_text(encoding="latin-1")

        sample = raw_text[:2048]
        delimiter = "\t" if dataset_path.suffix.lower() in {".tsv", ".txt"} else ","
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            delimiter = dialect.delimiter
        except csv.Error:
            pass

        reader = csv.DictReader(raw_text.splitlines(), delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("The selected file does not contain a header row.")

        points: list[tuple[float, float]] = []
        for row in reader:
            cycle_text = self._pick_row_value(
                row,
                ("Cycle Count", "Cycles", "Cycle", "cycle_count"),
            )
            absolute_length_text = self._pick_row_value(
                row,
                (
                    "Absolute Crack Length [mm]",
                    "Absolute [mm]",
                    "Crack Length [mm]",
                    "Crack Length",
                    "Length [mm]",
                ),
            )
            signed_length_text = self._pick_row_value(
                row,
                ("Signed Crack Length [mm]", "Signed [mm]", "Signed Crack Length"),
            )

            if cycle_text is None:
                continue
            if absolute_length_text is None and signed_length_text is None:
                continue

            try:
                cycle_count = float(cycle_text)
                if absolute_length_text is not None:
                    crack_length = float(absolute_length_text)
                else:
                    crack_length = abs(float(signed_length_text))
            except ValueError:
                continue

            points.append((cycle_count, crack_length))

        if not points:
            raise ValueError(
                "No usable data points were found. Expected cycle-count and crack-length columns."
            )

        points.sort(key=lambda point: point[0])
        return PlotDataset(name=dataset_path.stem, points=points, source_path=dataset_path)

    def import_graph_datasets(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import crack-growth data",
            str(Path.cwd()),
            "Delimited data (*.csv *.tsv *.txt);;CSV files (*.csv);;Tab-separated files (*.tsv *.txt)",
        )
        if not paths:
            return

        loaded_datasets: list[PlotDataset] = []
        failures: list[str] = []
        for path_string in paths:
            dataset_path = Path(path_string)
            try:
                dataset = self.load_plot_dataset_from_file(dataset_path)
            except (OSError, ValueError) as exc:
                failures.append(f"{dataset_path.name}: {exc}")
                continue

            replaced = False
            for index, existing in enumerate(self.imported_plot_datasets):
                if existing.source_path == dataset.source_path:
                    self.imported_plot_datasets[index] = dataset
                    replaced = True
                    break
            if not replaced:
                self.imported_plot_datasets.append(dataset)
            loaded_datasets.append(dataset)

        self.refresh_crack_growth_chart()
        self.autosave_session()

        if loaded_datasets:
            self.statusBar().showMessage(
                f"Imported {len(loaded_datasets)} crack-growth dataset(s).",
                5000,
            )
        if failures:
            QMessageBox.warning(
                self,
                "Some files could not be imported",
                "\n".join(failures),
            )

    def clear_imported_graph_datasets(self) -> None:
        self.imported_plot_datasets.clear()
        self.refresh_crack_growth_chart()
        self.autosave_session()
        self.statusBar().showMessage("Cleared imported crack-growth datasets.", 4000)

    @staticmethod
    def _axis_range(values: list[float], padding_ratio: float, minimum_span: float) -> tuple[float, float]:
        if not values:
            return 0.0, 1.0
        minimum_value = min(values)
        maximum_value = max(values)
        span = maximum_value - minimum_value
        if span < minimum_span:
            center = 0.5 * (minimum_value + maximum_value)
            half_span = 0.5 * minimum_span
            return center - half_span, center + half_span
        padding = span * padding_ratio
        return minimum_value - padding, maximum_value + padding

    def refresh_crack_growth_chart(self) -> None:
        chart = QChart()
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        chart.setTitle("Crack Length vs Cycle Count")
        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)

        datasets: list[PlotDataset] = []
        current_dataset = self.current_measurement_plot_dataset()
        if current_dataset.points:
            datasets.append(current_dataset)
        datasets.extend(dataset for dataset in self.imported_plot_datasets if dataset.points)

        x_values: list[float] = []
        y_values: list[float] = []
        for index, dataset in enumerate(datasets):
            color = CHART_SERIES_COLORS[index % len(CHART_SERIES_COLORS)]
            series = QLineSeries()
            series.setName(dataset.name)
            pen = QPen(color)
            pen.setWidth(2 if index == 0 else 3)
            series.setPen(pen)
            series.setColor(color)
            series.setPointsVisible(True)

            for x_value, y_value in dataset.points:
                series.append(float(x_value), float(y_value))
                x_values.append(float(x_value))
                y_values.append(float(y_value))

            chart.addSeries(series)

        axis_x = QValueAxis()
        axis_x.setTitleText("Cycle count")
        axis_x.setLabelFormat("%.0f")
        axis_y = QValueAxis()
        axis_y.setTitleText("Crack length [mm]")
        axis_y.setLabelFormat("%.3f")

        x_min, x_max = self._axis_range(x_values, padding_ratio=0.05, minimum_span=1.0)
        y_min, y_max = self._axis_range(y_values, padding_ratio=0.08, minimum_span=0.1)
        axis_x.setRange(x_min, x_max)
        axis_y.setRange(y_min, y_max)

        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        for series in chart.series():
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)

        if not datasets:
            chart.legend().setVisible(False)
            self.graph_tab.status_label.setText(
                "No crack-growth points are available yet. Measure images or import a CSV/TSV file to populate the graph."
            )
        else:
            imported_count = len([dataset for dataset in self.imported_plot_datasets if dataset.points])
            self.graph_tab.status_label.setText(
                f"Showing {len(current_dataset.points)} current-session point(s) and {imported_count} imported dataset(s)."
            )

        self.graph_tab.chart_view.setChart(chart)

    def apply_settings(self) -> None:
        settings = self.current_settings()
        if self.current_measurement_index is not None:
            self.load_measurement_record(self.current_measurement_index)

        if self.latest_live_original_bgr is not None:
            self.reprocess_live_frame(self.latest_live_original_bgr, settings)

        if self.camera_worker is not None:
            self.camera_worker.update_settings(settings)

        self.refresh_crack_growth_chart()
        self.autosave_session()
        self.statusBar().showMessage("Settings applied.", 4000)

    def reset_settings(self) -> None:
        self.settings_panel.set_settings(AppSettings())
        self.apply_settings()

    def save_profile(self) -> None:
        profile_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save settings profile",
            str(Path.cwd() / "aruco_profile.json"),
            "JSON files (*.json)",
        )
        if not profile_path:
            return

        settings = self.current_settings()
        Path(profile_path).write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
        self.statusBar().showMessage(f"Saved profile to {profile_path}", 4000)

    def load_profile(self) -> None:
        profile_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load settings profile",
            str(Path.cwd()),
            "JSON files (*.json)",
        )
        if not profile_path:
            return

        try:
            data = json.loads(Path(profile_path).read_text(encoding="utf-8"))
            settings = AppSettings.from_dict(data)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Profile load failed", str(exc))
            return

        self.settings_panel.set_settings(settings)
        self.apply_settings()

    def load_measurement_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select images",
            str(Path.cwd()),
            "Images (*.bmp *.jpeg *.jpg *.png *.tif *.tiff)",
        )
        if not paths:
            return
        self.load_measurement_records([Path(path) for path in paths])

    def load_measurement_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select an image folder", str(Path.cwd()))
        if not folder:
            return

        image_paths = list_image_files(Path(folder))
        if not image_paths:
            QMessageBox.information(self, "No images found", "The selected folder does not contain supported image files.")
            return
        self.load_measurement_records(image_paths)

    def load_measurement_records(self, image_paths: list[Path]) -> None:
        self.measurement_records = [MeasurementRecord(image_path=path) for path in image_paths]
        self.current_measurement_index = None
        self.refresh_measurement_table()
        if self.measurement_records:
            self.select_measurement_row(0)
        self.refresh_crack_growth_chart()
        self.autosave_session()
        self.statusBar().showMessage(f"Loaded {len(self.measurement_records)} images.", 4000)

    def refresh_measurement_table(self) -> None:
        self._updating_measurement_table = True
        table = self.measurement_tab.table
        table.setRowCount(len(self.measurement_records))
        for row, record in enumerate(self.measurement_records):
            self._set_table_item(row, 0, record.image_path.name)
            self._set_table_item(row, 1, str(record.cycle_count))
            self._set_table_item(row, 2, self._format_float(record.saved_absolute_length_mm))
            self._set_table_item(row, 3, self._format_float(record.saved_signed_length_mm))
            self._set_table_item(row, 4, record.status)
        self._updating_measurement_table = False

    def _set_table_item(self, row: int, column: int, text: str) -> None:
        item = QTableWidgetItem(text)
        if column != 0:
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.measurement_tab.table.setItem(row, column, item)

    @staticmethod
    def _format_float(value: float | None) -> str:
        if value is None:
            return ""
        return f"{value:.3f}"

    def select_measurement_row(self, index: int) -> None:
        if not (0 <= index < len(self.measurement_records)):
            return
        self.measurement_tab.table.blockSignals(True)
        self.measurement_tab.table.selectRow(index)
        self.measurement_tab.table.blockSignals(False)
        self.load_measurement_record(index)

    def on_measurement_selection_changed(self) -> None:
        if self._updating_measurement_table:
            return
        selected_items = self.measurement_tab.table.selectedItems()
        if not selected_items:
            return
        self.load_measurement_record(selected_items[0].row())

    def load_measurement_record(self, index: int) -> None:
        if not (0 <= index < len(self.measurement_records)):
            return

        record = self.measurement_records[index]
        self.current_measurement_index = index
        self.measurement_tab.current_file_label.setText(f"Current image: {record.image_path.name}")
        self.measurement_tab.cycle_spin.blockSignals(True)
        self.measurement_tab.cycle_spin.setValue(record.cycle_count)
        self.measurement_tab.cycle_spin.blockSignals(False)
        self.current_provisional_measurement = None

        try:
            image_bgr = load_color_image(record.image_path)
            rectification = detect_and_rectify(
                image_bgr,
                self.current_settings().detector,
                self.current_settings().calibration,
            )
            self.current_measurement_image_bgr = image_bgr
            self.current_measurement_rectification = rectification
            if record.saved_tip_mm is not None:
                self.current_provisional_measurement = measure_crack_from_marker_point(
                    record.saved_tip_mm,
                    rectification.marker_corners_rectified_px,
                    self.current_settings().calibration,
                )
            self.render_measurement_views()
            self.measurement_tab.status_label.setText(
                (
                    f"Detected marker {rectification.marker_id} "
                    f"({rectification.detected_marker_count} candidate(s), "
                    f"{rectification.rejected_candidate_count} rejected, "
                    f"{rectification.corner_source})."
                )
            )
        except ProcessingError as exc:
            self.current_measurement_image_bgr = None
            self.current_measurement_rectification = None
            self.current_provisional_measurement = None
            self.measurement_tab.original_canvas.clear()
            self.measurement_tab.rectified_canvas.clear()
            self.measurement_tab.status_label.setText(str(exc))

        self.update_measurement_labels()

    def render_measurement_views(self) -> None:
        original_rgb = bgr_to_rgb(self.current_measurement_image_bgr)
        rectified_rgb = (
            bgr_to_rgb(self.current_measurement_rectification.rectified_bgr)
            if self.current_measurement_rectification is not None
            else None
        )
        self.measurement_tab.original_canvas.set_image(original_rgb)
        self.measurement_tab.rectified_canvas.set_image(rectified_rgb)
        self.measurement_tab.original_canvas.set_overlay(self.build_original_overlay())
        self.measurement_tab.rectified_canvas.set_overlay(self.build_rectified_overlay())

    def build_original_overlay(self) -> CanvasOverlay:
        return self._build_original_measurement_overlay(
            self.current_measurement_rectification,
            self.current_provisional_measurement,
        )

    def _build_original_measurement_overlay(
        self,
        rectification,
        measurement,
    ) -> CanvasOverlay:
        overlay = CanvasOverlay()
        if rectification is None:
            return overlay

        overlay.polygons.append(
            OverlayPolygon(
                points=[tuple(map(float, point)) for point in rectification.marker_corners_image_px],
                color=MARKER_OVERLAY_COLOR,
                width=2,
            )
        )
        if measurement is not None:
            original_top_pin = project_rectified_point_to_image(
                measurement.top_pin_rectified_px,
                rectification,
            )
            original_bottom_pin = project_rectified_point_to_image(
                measurement.bottom_pin_rectified_px,
                rectification,
            )
            original_projection = project_rectified_point_to_image(
                measurement.load_line_projection_rectified_px,
                rectification,
            )
            original_tip = project_rectified_point_to_image(
                measurement.crack_tip_rectified_px,
                rectification,
            )
            overlay.points.extend(
                [
                    OverlayPoint(
                        x=float(original_top_pin[0]),
                        y=float(original_top_pin[1]),
                        color=PIN_OVERLAY_COLOR,
                        radius=6,
                        cross=False,
                    ),
                    OverlayPoint(
                        x=float(original_bottom_pin[0]),
                        y=float(original_bottom_pin[1]),
                        color=PIN_OVERLAY_COLOR,
                        radius=6,
                        cross=False,
                    ),
                    OverlayPoint(
                        x=float(original_projection[0]),
                        y=float(original_projection[1]),
                        color=CRACK_OVERLAY_COLOR,
                        radius=5,
                        cross=False,
                    ),
                    OverlayPoint(
                        x=float(original_tip[0]),
                        y=float(original_tip[1]),
                        color=CRACK_OVERLAY_COLOR,
                        radius=8,
                        label=f"{measurement.absolute_length_mm:.3f} mm",
                    ),
                ]
            )
            overlay.lines.extend(
                [
                    OverlayLine(
                        start=tuple(map(float, original_top_pin)),
                        end=tuple(map(float, original_bottom_pin)),
                        color=PIN_OVERLAY_COLOR,
                        width=2,
                    ),
                    OverlayLine(
                        start=tuple(map(float, original_projection)),
                        end=tuple(map(float, original_tip)),
                        color=CRACK_OVERLAY_COLOR,
                        width=2,
                    ),
                ]
            )
        return overlay

    def build_rectified_overlay(self) -> CanvasOverlay:
        return self._build_rectified_measurement_overlay(
            self.current_measurement_rectification,
            self.current_provisional_measurement,
        )

    def _build_rectified_measurement_overlay(
        self,
        rectification,
        measurement,
    ) -> CanvasOverlay:
        overlay = CanvasOverlay()
        if rectification is None:
            return overlay

        overlay.polygons.append(
            OverlayPolygon(
                points=[tuple(map(float, point)) for point in rectification.marker_corners_rectified_px],
                color=MARKER_OVERLAY_COLOR,
                width=2,
            )
        )

        if measurement is not None:
            overlay.points.extend(
                [
                    OverlayPoint(
                        x=float(measurement.top_pin_rectified_px[0]),
                        y=float(measurement.top_pin_rectified_px[1]),
                        color=PIN_OVERLAY_COLOR,
                        radius=7,
                        label="Top pin",
                    ),
                    OverlayPoint(
                        x=float(measurement.bottom_pin_rectified_px[0]),
                        y=float(measurement.bottom_pin_rectified_px[1]),
                        color=PIN_OVERLAY_COLOR,
                        radius=7,
                        label="Bottom pin",
                    ),
                    OverlayPoint(
                        x=float(measurement.load_line_projection_rectified_px[0]),
                        y=float(measurement.load_line_projection_rectified_px[1]),
                        color=CRACK_OVERLAY_COLOR,
                        radius=5,
                        cross=False,
                    ),
                    OverlayPoint(
                        x=float(measurement.crack_tip_rectified_px[0]),
                        y=float(measurement.crack_tip_rectified_px[1]),
                        color=CRACK_OVERLAY_COLOR,
                        radius=8,
                        label=f"{measurement.absolute_length_mm:.3f} mm",
                    ),
                ]
            )
            overlay.lines.extend(
                [
                    OverlayLine(
                        start=tuple(map(float, measurement.top_pin_rectified_px)),
                        end=tuple(map(float, measurement.bottom_pin_rectified_px)),
                        color=PIN_OVERLAY_COLOR,
                        width=2,
                    ),
                    OverlayLine(
                        start=tuple(map(float, measurement.load_line_projection_rectified_px)),
                        end=tuple(map(float, measurement.crack_tip_rectified_px)),
                        color=CRACK_OVERLAY_COLOR,
                        width=2,
                    ),
                ]
            )
        return overlay

    def update_measurement_labels(self) -> None:
        measurement = self.current_provisional_measurement
        if measurement is None:
            self.measurement_tab.absolute_length_label.setText("Absolute length: --")
            self.measurement_tab.signed_length_label.setText("Signed length: --")
            self.measurement_tab.tip_position_label.setText("Crack tip [mm]: --")
            return

        self.measurement_tab.absolute_length_label.setText(
            f"Absolute length: {measurement.absolute_length_mm:.3f} mm"
        )
        self.measurement_tab.signed_length_label.setText(
            f"Signed length: {measurement.signed_length_mm:.3f} mm"
        )
        self.measurement_tab.tip_position_label.setText(
            f"Crack tip [mm]: x={measurement.crack_tip_mm[0]:.3f}, y={measurement.crack_tip_mm[1]:.3f}"
        )

    def handle_measurement_original_click(self, x: float, y: float) -> None:
        if self.current_measurement_rectification is None:
            return
        try:
            self.current_provisional_measurement = measure_crack_from_original_point(
                np.array([x, y], dtype=np.float64),
                self.current_measurement_rectification,
                self.current_settings().calibration,
            )
            self.render_measurement_views()
            self.update_measurement_labels()
            self.store_current_measurement_record()
            self.refresh_crack_growth_chart()
            self.autosave_session()
            self.measurement_tab.status_label.setText("Updated crack-tip measurement from the original image.")
        except ProcessingError as exc:
            self.measurement_tab.status_label.setText(str(exc))

    def handle_measurement_rectified_click(self, x: float, y: float) -> None:
        if self.current_measurement_rectification is None:
            return
        try:
            self.current_provisional_measurement = measure_crack_from_rectified_point(
                np.array([x, y], dtype=np.float64),
                self.current_measurement_rectification.marker_corners_rectified_px,
                self.current_settings().calibration,
            )
            self.render_measurement_views()
            self.update_measurement_labels()
            self.store_current_measurement_record()
            self.refresh_crack_growth_chart()
            self.autosave_session()
            self.measurement_tab.status_label.setText("Updated crack-tip measurement from the rectified image.")
        except ProcessingError as exc:
            self.measurement_tab.status_label.setText(str(exc))

    def reprocess_current_measurement(self) -> None:
        if self.current_measurement_index is None:
            return
        self.load_measurement_record(self.current_measurement_index)

    def save_current_measurement(self) -> None:
        if self.current_measurement_index is None:
            return
        if not self.store_current_measurement_record(require_measurement=True):
            return
        record = self.measurement_records[self.current_measurement_index]
        self.refresh_crack_growth_chart()
        self.autosave_session()
        self.statusBar().showMessage(
            f"Saved measurement for {record.image_path.name} and updated the autosave session.",
            5000,
        )

    def handle_cycle_count_changed(self, value: int) -> None:
        if self.current_measurement_index is None:
            return

        self.measurement_records[self.current_measurement_index].cycle_count = value
        self.refresh_measurement_row(self.current_measurement_index)
        self.refresh_crack_growth_chart()
        self.autosave_session()

    def export_measurements(self) -> None:
        if not self.measurement_records:
            QMessageBox.information(self, "Nothing to export", "Load and measure images first.")
            return

        export_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export measurements",
            str(Path.cwd() / "measurements.tsv"),
            "Tab-separated values (*.tsv *.txt)",
        )
        if not export_path:
            return

        with Path(export_path).open("w", encoding="utf-8", newline="") as file_handle:
            writer = csv.writer(file_handle, delimiter="\t")
            writer.writerow(
                [
                    "Image Name",
                    "Cycle Count",
                    "Absolute Crack Length [mm]",
                    "Signed Crack Length [mm]",
                    "Crack Tip X [mm]",
                    "Crack Tip Y [mm]",
                    "Marker ID",
                    "Status",
                ]
            )
            for record in self.measurement_records:
                tip_x = ""
                tip_y = ""
                if record.saved_tip_mm is not None:
                    tip_x = f"{float(record.saved_tip_mm[0]):.3f}"
                    tip_y = f"{float(record.saved_tip_mm[1]):.3f}"
                writer.writerow(
                    [
                        record.image_path.name,
                        record.cycle_count,
                        self._format_float(record.saved_absolute_length_mm),
                        self._format_float(record.saved_signed_length_mm),
                        tip_x,
                        tip_y,
                        record.marker_id if record.marker_id is not None else "",
                        record.status,
                    ]
                )

        self.statusBar().showMessage(f"Exported measurements to {export_path}", 4000)

    def select_previous_measurement(self) -> None:
        if self.current_measurement_index is None:
            return
        self.select_measurement_row(max(0, self.current_measurement_index - 1))

    def select_next_measurement(self) -> None:
        if self.current_measurement_index is None:
            return
        self.select_measurement_row(
            min(len(self.measurement_records) - 1, self.current_measurement_index + 1)
        )

    def start_camera(self) -> None:
        if self.camera_thread is not None:
            return

        from PySide6.QtCore import QThread

        self.camera_thread = QThread(self)
        self.camera_worker = CameraWorker(self.current_settings())
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.frame_ready.connect(self.handle_live_frame)
        self.camera_worker.finished.connect(self.camera_thread.quit)
        self.camera_worker.finished.connect(self.camera_worker.deleteLater)
        self.camera_thread.finished.connect(self.on_camera_finished)
        self.camera_thread.start()
        self.live_tab.status_label.setText("Starting camera...")
        self.statusBar().showMessage("Starting live camera stream.", 4000)

    def stop_camera(self) -> None:
        if self.camera_worker is None:
            return
        self.camera_worker.stop()

    def on_camera_finished(self) -> None:
        if self.camera_thread is not None:
            self.camera_thread.deleteLater()
        self.camera_thread = None
        self.camera_worker = None
        self.live_tab.status_label.setText("Camera is idle.")

    def handle_live_frame(self, frame_result) -> None:
        self.latest_live_original_bgr = frame_result.original_bgr
        self.latest_live_rectification = frame_result.rectification
        self.reprocess_live_frame(
            frame_result.original_bgr,
            self.current_settings(),
            error_override=frame_result.error,
            rectification_override=frame_result.rectification,
        )

    def reprocess_live_frame(
        self,
        original_bgr,
        settings: AppSettings,
        error_override: str | None = None,
        rectification_override=None,
    ) -> None:
        if original_bgr is None:
            self.live_tab.original_canvas.clear()
            self.live_tab.rectified_canvas.clear()
            self.live_tab.status_label.setText(error_override or "No live frame available.")
            return

        rectification = rectification_override
        error_message = error_override
        if rectification is None and error_override is None:
            try:
                rectification = detect_and_rectify(
                    original_bgr,
                    settings.detector,
                    settings.calibration,
                )
                error_message = None
            except ProcessingError as exc:
                rectification = None
                error_message = str(exc)

        self.latest_live_rectification = rectification
        self.live_tab.original_canvas.set_image(bgr_to_rgb(original_bgr))
        self.live_tab.rectified_canvas.set_image(
            bgr_to_rgb(rectification.rectified_bgr) if rectification is not None else None
        )

        live_measurement = None
        if rectification is not None and self.live_tip_mm is not None:
            try:
                live_measurement = measure_crack_from_marker_point(
                    self.live_tip_mm,
                    rectification.marker_corners_rectified_px,
                    settings.calibration,
                )
            except ProcessingError as exc:
                error_message = str(exc)

        self.live_tab.original_canvas.set_overlay(
            self.build_live_original_overlay(rectification, live_measurement)
        )
        self.live_tab.rectified_canvas.set_overlay(
            self.build_live_rectified_overlay(rectification, live_measurement)
        )
        self.update_live_labels(rectification, live_measurement)
        self.live_tab.status_label.setText(
            error_message
            or (
                f"Detected marker {rectification.marker_id} in the live stream "
                f"using {rectification.corner_source}."
                if rectification is not None
                else "Waiting for a detectable marker."
            )
        )

    def build_live_original_overlay(self, rectification, live_measurement) -> CanvasOverlay:
        return self._build_original_measurement_overlay(rectification, live_measurement)

    def build_live_rectified_overlay(self, rectification, live_measurement) -> CanvasOverlay:
        return self._build_rectified_measurement_overlay(rectification, live_measurement)

    def update_live_labels(self, rectification, live_measurement) -> None:
        if rectification is not None:
            self.live_tab.marker_label.setText(
                f"Marker: {rectification.marker_id} ({rectification.corner_source})"
            )
        else:
            self.live_tab.marker_label.setText("Marker: --")

        if live_measurement is None:
            self.live_tab.absolute_length_label.setText("Absolute length: --")
            self.live_tab.signed_length_label.setText("Signed length: --")
            self.live_tab.tip_position_label.setText("Crack tip [mm]: --")
            return

        self.live_tab.absolute_length_label.setText(
            f"Absolute length: {live_measurement.absolute_length_mm:.3f} mm"
        )
        self.live_tab.signed_length_label.setText(
            f"Signed length: {live_measurement.signed_length_mm:.3f} mm"
        )
        self.live_tab.tip_position_label.setText(
            f"Crack tip [mm]: x={live_measurement.crack_tip_mm[0]:.3f}, y={live_measurement.crack_tip_mm[1]:.3f}"
        )

    def handle_live_rectified_click(self, x: float, y: float) -> None:
        if self.latest_live_rectification is None:
            return
        try:
            measurement = measure_crack_from_rectified_point(
                np.array([x, y], dtype=np.float64),
                self.latest_live_rectification.marker_corners_rectified_px,
                self.current_settings().calibration,
            )
        except ProcessingError as exc:
            self.live_tab.status_label.setText(str(exc))
            return

        self.live_tip_mm = measurement.crack_tip_mm.copy()
        self.reprocess_live_frame(self.latest_live_original_bgr, self.current_settings())
        self.autosave_session()
        self.live_tab.status_label.setText("Updated the live crack-tip tracking point.")

    def clear_live_tip(self) -> None:
        self.live_tip_mm = None
        if self.latest_live_original_bgr is not None:
            self.reprocess_live_frame(self.latest_live_original_bgr, self.current_settings())
        self.autosave_session()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.store_current_measurement_record()
        self.autosave_session()
        self.stop_camera()
        super().closeEvent(event)


def launch_app() -> int:
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication([])
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    if owns_app:
        return app.exec()
    return 0

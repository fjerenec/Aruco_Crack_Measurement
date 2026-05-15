from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget


@dataclass
class OverlayPoint:
    x: float
    y: float
    color: tuple[int, ...] = (255, 80, 80, 190)
    radius: int = 6
    cross: bool = True
    label: str = ""


@dataclass
class OverlayLine:
    start: tuple[float, float]
    end: tuple[float, float]
    color: tuple[int, ...] = (255, 80, 80, 190)
    width: int = 2
    dashed: bool = False
    label: str = ""


@dataclass
class OverlayPolygon:
    points: list[tuple[float, float]]
    color: tuple[int, ...] = (255, 215, 0, 175)
    width: int = 2
    dashed: bool = False


@dataclass
class CanvasOverlay:
    points: list[OverlayPoint] = field(default_factory=list)
    lines: list[OverlayLine] = field(default_factory=list)
    polygons: list[OverlayPolygon] = field(default_factory=list)


class ImageCanvas(QWidget):
    clicked = Signal(float, float)

    def __init__(self, placeholder_text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._placeholder_text = placeholder_text
        self._overlay = CanvasOverlay()
        self._pixmap: QPixmap | None = None
        self._image_width = 0
        self._image_height = 0
        self._image_rect = QRectF()

    def set_placeholder_text(self, text: str) -> None:
        self._placeholder_text = text
        self.update()

    def set_image(self, image_rgb: np.ndarray | None) -> None:
        if image_rgb is None:
            self._pixmap = None
            self._image_width = 0
            self._image_height = 0
            self._image_rect = QRectF()
            self.update()
            return

        image_rgb = np.ascontiguousarray(image_rgb)
        image_height, image_width = image_rgb.shape[:2]
        bytes_per_line = image_width * image_rgb.shape[2]
        qimage = QImage(
            image_rgb.data,
            image_width,
            image_height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        self._pixmap = QPixmap.fromImage(qimage)
        self._image_width = image_width
        self._image_height = image_height
        self.update()

    def set_overlay(self, overlay: CanvasOverlay | None) -> None:
        self._overlay = overlay or CanvasOverlay()
        self.update()

    def clear(self) -> None:
        self.set_image(None)
        self.set_overlay(None)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        image_point = self._widget_to_image(event.position())
        if image_point is not None:
            self.clicked.emit(image_point.x(), image_point.y())
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if self._pixmap is None:
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder_text)
            return

        self._image_rect = self._compute_image_rect()
        painter.drawPixmap(self._image_rect.toRect(), self._pixmap)
        self._draw_overlay(painter)
        painter.end()

    def _compute_image_rect(self) -> QRectF:
        if self._pixmap is None:
            return QRectF()

        widget_width = max(1, self.width())
        widget_height = max(1, self.height())
        image_aspect = self._image_width / max(1, self._image_height)
        widget_aspect = widget_width / max(1, widget_height)

        if image_aspect > widget_aspect:
            draw_width = widget_width
            draw_height = draw_width / image_aspect
        else:
            draw_height = widget_height
            draw_width = draw_height * image_aspect

        x_offset = (widget_width - draw_width) / 2.0
        y_offset = (widget_height - draw_height) / 2.0
        return QRectF(x_offset, y_offset, draw_width, draw_height)

    def _image_to_widget(self, x: float, y: float) -> QPointF:
        if self._image_rect.isNull():
            return QPointF()

        x_ratio = x / max(1, self._image_width)
        y_ratio = y / max(1, self._image_height)
        return QPointF(
            self._image_rect.left() + x_ratio * self._image_rect.width(),
            self._image_rect.top() + y_ratio * self._image_rect.height(),
        )

    def _widget_to_image(self, point: QPointF) -> QPointF | None:
        if self._pixmap is None or not self._image_rect.contains(point):
            return None

        x_ratio = (point.x() - self._image_rect.left()) / self._image_rect.width()
        y_ratio = (point.y() - self._image_rect.top()) / self._image_rect.height()
        return QPointF(
            x_ratio * self._image_width,
            y_ratio * self._image_height,
        )

    def _draw_overlay(self, painter: QPainter) -> None:
        for polygon in self._overlay.polygons:
            pen = QPen(self._to_qcolor(polygon.color))
            pen.setWidth(polygon.width)
            pen.setStyle(Qt.PenStyle.DashLine if polygon.dashed else Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            if len(polygon.points) < 2:
                continue

            widget_points = [self._image_to_widget(x, y) for x, y in polygon.points]
            for index, start_point in enumerate(widget_points):
                end_point = widget_points[(index + 1) % len(widget_points)]
                painter.drawLine(start_point, end_point)

        for line in self._overlay.lines:
            pen = QPen(self._to_qcolor(line.color))
            pen.setWidth(line.width)
            pen.setStyle(Qt.PenStyle.DashLine if line.dashed else Qt.PenStyle.SolidLine)
            painter.setPen(pen)

            start_point = self._image_to_widget(*line.start)
            end_point = self._image_to_widget(*line.end)
            painter.drawLine(start_point, end_point)
            if line.label:
                self._draw_label(painter, end_point, line.label)

        for point in self._overlay.points:
            pen = QPen(self._to_qcolor(point.color))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            center = self._image_to_widget(point.x, point.y)
            if point.cross:
                painter.drawLine(
                    QPointF(center.x() - point.radius, center.y() - point.radius),
                    QPointF(center.x() + point.radius, center.y() + point.radius),
                )
                painter.drawLine(
                    QPointF(center.x() - point.radius, center.y() + point.radius),
                    QPointF(center.x() + point.radius, center.y() - point.radius),
                )
            else:
                painter.drawEllipse(center, point.radius, point.radius)
            if point.label:
                self._draw_label(painter, center, point.label)

    @staticmethod
    def _to_qcolor(color: tuple[int, ...]) -> QColor:
        if len(color) == 4:
            return QColor(color[0], color[1], color[2], color[3])
        return QColor(color[0], color[1], color[2])

    def _draw_label(self, painter: QPainter, anchor: QPointF, text: str) -> None:
        text_rect = painter.fontMetrics().boundingRect(text).adjusted(-6, -4, 6, 4)
        target_x = min(max(0, int(anchor.x()) + 12), max(0, self.width() - text_rect.width()))
        target_y = min(
            max(0, int(anchor.y()) - text_rect.height() - 10),
            max(0, self.height() - text_rect.height()),
        )
        text_rect.moveTopLeft(QPoint(target_x, target_y))
        painter.fillRect(text_rect, QColor(0, 0, 0, 170))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, text)

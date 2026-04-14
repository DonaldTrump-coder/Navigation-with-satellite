from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt
import rasterio
import numpy as np
from PyQt6.QtGui import QImage, QPixmap

class SatelliteLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.min_lon = None
        self.max_lon = None
        self.min_lat = None
        self.max_lat = None
        self.img = None # original image array
        self.display_img = None # image array to display
        
    def load_image(self, tiff_path):
        with rasterio.open(tiff_path) as src:
            img = src.read()
            bounds = src.bounds
            self.min_lon = bounds.left
            self.max_lon = bounds.right
            self.min_lat = bounds.bottom
            self.max_lat = bounds.top
            self.img = img
        
        if img.shape[0] >= 3:
            rgb = np.stack([img[0], img[1], img[2]], axis=-1)
        else:
            gray = img[0]
            rgb = np.stack([gray, gray, gray], axis=-1)
        rgb = rgb.astype(np.float32)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
        rgb = (rgb * 255).astype(np.uint8)
        self.display_img = rgb
        
        h, w, ch = rgb.shape
        qimg = QImage(
            rgb.data,
            w,
            h,
            ch * w,
            QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimg)
        self.setPixmap(pixmap.scaled(
            self.width(),
            self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
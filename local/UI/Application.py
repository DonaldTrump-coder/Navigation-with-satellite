from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QScrollArea, QLineEdit, QPushButton
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl, Qt, QThread, QTimer, pyqtSignal
from local.UI.SatelliteLabel import SatelliteLabel
from local.controller.UAVcontroller import UAVContoller

class Application(QMainWindow):
    stop_signal = pyqtSignal() # stop of UAV controller
    def __init__(self, address):
        super().__init__()
        self.setWindowTitle("UE-Satellite Navigator")
        self.resize(1000, 600)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout()
        self.UE_browser = QWebEngineView()
        self.UE_browser.load(QUrl(address))
        right_widget = QWidget()
        main_layout.addWidget(self.UE_browser, 2)
        main_layout.addWidget(right_widget, 1)
        main_widget.setLayout(main_layout)
        
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        
        # log area
        self.log_area = QScrollArea()
        self.log_area.setWidgetResizable(True)
        self.log_container = QWidget()
        self.log_layout = QVBoxLayout()
        self.log_layout.addStretch()
        self.log_container.setLayout(self.log_layout)
        self.log_area.setWidget(self.log_container)
        
        # input area
        input_widget = QWidget()
        input_layout = QHBoxLayout()
        input_widget.setLayout(input_layout)
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Input Tasks...")
        self.input_button = QPushButton("Excecute")
        input_layout.addWidget(self.input_box)
        input_layout.addWidget(self.input_button)
        
        # satellite image
        self.image_label = SatelliteLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        right_layout.addWidget(self.log_area, 4)
        right_layout.addWidget(input_widget, 0)
        right_layout.addWidget(self.image_label, 3)
        
        QTimer.singleShot(0, self.start_control)
        
    def start_control(self):
        self.UAV_thread = QThread()
        self.controller = UAVContoller()
        self.controller.moveToThread(self.UAV_thread)
        self.UAV_thread.started.connect(self.controller.init_drone)
        self.stop_signal.connect(self.controller.stop)
        self.controller.position_signal.connect(self.update_position)
        self.UAV_thread.start()
        
    def keyPressEvent(self, event):
        if not self.controller:
            return
        key = event.key()
        
        if key == Qt.Key.Key_W:
            self.controller.forward()
        elif key == Qt.Key.Key_S:
            self.controller.backward()
        elif key == Qt.Key.Key_A:
            self.controller.move_left()
        elif key == Qt.Key.Key_D:
            self.controller.move_right()
        elif key == Qt.Key.Key_Q:
            self.controller.turn_left()
        elif key == Qt.Key.Key_E:
            self.controller.turn_right()
        elif key == Qt.Key.Key_Space:
            self.controller.up()
        elif key == Qt.Key.Key_Shift:
            self.controller.down()
            
    def keyReleaseEvent(self, event):
        if not self.controller:
            return
        self.controller.unlease()
        
    def closeEvent(self, event):
        if self.controller:
            self.stop_signal.emit()
        if self.UAV_thread:
            self.UAV_thread.quit()
            
        event.accept()
# visualizer_base.py

from abc import abstractmethod
from PyQt5 import QtCore, QtWidgets
from audioviz.audio_processing.audio_processor import AudioProcessor

class VisualizerBase(QtWidgets.QWidget):
    def __init__(self, processor: AudioProcessor, update_interval_ms: int = 50, parent=None):
        super().__init__(parent)
        self.processor = processor

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(update_interval_ms)
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start()

    @abstractmethod
    def update_visualization(self):
        """Update the visualization based on new processor data."""
        pass

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton

class WavParamDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ultrasonic WAV Parameters")
        self.setFixedSize(300, 150)

        layout = QVBoxLayout()

        # Frequency
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Frequency (Hz):"))
        self.freq_edit = QLineEdit("19000")
        freq_layout.addWidget(self.freq_edit)
        layout.addLayout(freq_layout)

        # Duration
        dur_layout = QHBoxLayout()
        dur_layout.addWidget(QLabel("Duration (s):"))
        self.dur_edit = QLineEdit("0.1")
        dur_layout.addWidget(self.dur_edit)
        layout.addLayout(dur_layout)

        # Sample Rate
        fs_layout = QHBoxLayout()
        fs_layout.addWidget(QLabel("Sample Rate (Hz):"))
        self.fs_edit = QLineEdit("48000")
        fs_layout.addWidget(self.fs_edit)
        layout.addLayout(fs_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def get_values(self):
        # Return input parameters (freq, duration, fs), with default values and type conversion
        try:
            freq = float(self.freq_edit.text())
        except:
            freq = 19000
        try:
            duration = float(self.dur_edit.text())
        except:
            duration = 0.1
        try:
            fs = int(self.fs_edit.text())
        except:
            fs = 48000
        return freq, duration, fs

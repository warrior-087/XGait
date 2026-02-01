import socket
import threading
import json
import os
import struct
import numpy as np
import wave
import pyaudio
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel,
    QTableWidget, QTableWidgetItem, QHBoxLayout, QLineEdit, QFileDialog
)
from PyQt6.QtCore import pyqtSignal, QThread
from wave_params_dlg import WavParamDialog

# ===================== Utility Functions =====================

def get_local_ip():
    """Automatically get the local machine's IPv4 address in the LAN (no real connection needed).
    Returns 127.0.0.1 as fallback.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Not actually sending packets to 8.8.8.8, just letting the kernel choose the appropriate NIC
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        try:
            s.close()
        except:
            pass
    return ip

############################################
# Background Server Thread: Handle Client Connections and Commands
############################################
class ServerThread(QThread):
    # Dictionary to record client_name
    client_names = {}  # client_id -> client_name

    client_connected = pyqtSignal(str)
    client_disconnected = pyqtSignal(str)
    log = pyqtSignal(str)
    file_received = pyqtSignal(str, str)  # client_id, saved_path

    def __init__(self, host="0.0.0.0", port=5001):
        super().__init__()
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients = {}  # client_id -> socket
        self.running = True
        # Prefix modifiable by GUI (e.g., user2-1-1)
        self.filename_prefix = ""

    def run(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Bind to specified host (can be 0.0.0.0 or LAN IP)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.log.emit(f"Server listening on {self.host}:{self.port}")

            while self.running:
                client_socket, addr = self.server_socket.accept()
                client_id = f"{addr[0]}:{addr[1]}"
                self.clients[client_id] = client_socket
                self.log.emit(f"Client connected: {client_id}")
                self.client_connected.emit(client_id)

                thread = threading.Thread(target=self.handle_client, args=(client_socket, client_id), daemon=True)
                thread.start()
        except Exception as e:
            self.log.emit(f"Server thread error: {e}")

    def stop(self):
        self.running = False
        try:
            if self.server_socket:
                self.server_socket.close()
        except:
            pass

    def handle_client(self, client_socket, client_id):
        try:
            while True:
                # Receive a small header first, assume client sends decodable text command (like JSON or FILE_UPLOAD ...)
                header = client_socket.recv(4096)
                if not header:
                    break

                try:
                    msg = header.decode(errors="ignore").strip()
                except Exception:
                    msg = ""

                # If JSON, try to parse client_name first
                try:
                    obj = json.loads(msg)
                    if isinstance(obj, dict):
                        if "client_name" in obj:
                            name = obj["client_name"]
                            self.client_names[client_id] = name
                            self.log.emit(f"Received client_name from {client_id}: {name}")
                            continue
                        if "cmd" in obj:
                            cmd = obj["cmd"]
                            # Handle START_RECORD / STOP_RECORD or other commands
                            self.log.emit(f"JSON cmd from {client_id}: {cmd}")
                            continue
                except json.JSONDecodeError:
                    pass

                # File upload command (text): FILE_UPLOAD filename size
                if msg.startswith("FILE_UPLOAD"):
                    parts = msg.split()
                    if len(parts) < 3:
                        self.log.emit(f"Malformed FILE_UPLOAD from {client_id}: {msg}")
                        continue
                    _, filename, size = parts[:3]
                    try:
                        size = int(size)
                    except:
                        self.log.emit(f"Invalid size in FILE_UPLOAD from {client_id}: {size}")
                        continue
                    self.log.emit(f"Receiving file {filename} ({size} bytes) from {client_id}")

                    # Save based on client_name
                    client_name = self.client_names.get(client_id, client_id.replace(":", "_"))
                    save_dir = os.path.join("data", client_name)
                    os.makedirs(save_dir, exist_ok=True)

                    # Generate filename based on GUI specified prefix
                    prefix = self.filename_prefix.strip()
                    if prefix != "":
                        base, ext = os.path.splitext(filename)
                        if ext == "":
                            ext = ".wav"
                        new_name = f"{prefix}-{client_name}{ext}"
                    else:
                        new_name = f"{filename}-{client_name}"

                    filepath = os.path.join(save_dir, new_name)

                    # ip_only = client_id.split(":")[0]  # extract IP, no port
                    # save_dir = os.path.join("data", ip_only)
                    # os.makedirs(save_dir, exist_ok=True)
                    #
                    # # Decide final save name based on GUI specified prefix
                    # prefix = self.filename_prefix.strip()
                    # if prefix != "":
                    #     base, ext = os.path.splitext(filename)
                    #     if ext == "":
                    #         ext = ".wav"
                    #     new_name = prefix + ext
                    # else:
                    #     new_name = filename
                    #
                    # filepath = os.path.join(save_dir, new_name)

                    # Receive fixed amount of bytes
                    remaining = size
                    with open(filepath, "wb") as f:
                        while remaining > 0:
                            chunk = client_socket.recv(min(4096, remaining))
                            if not chunk:
                                break
                            f.write(chunk)
                            remaining -= len(chunk)

                    self.log.emit(f"Saved file to {filepath}")
                    self.file_received.emit(client_id, filepath)
                    continue

                # Others are JSON control messages (e.g. status report)
                try:
                    obj = json.loads(msg)
                    self.log.emit(f"JSON from {client_id}: {obj}")
                except Exception:
                    # Non-JSON message, log directly
                    if msg:
                        self.log.emit(f"Recv from {client_id}: {msg}")

        except Exception as e:
            self.log.emit(f"Client handler error {client_id}: {e}")

        # Handle disconnection
        try:
            if client_id in self.clients:
                del self.clients[client_id]
        except:
            pass
        self.log.emit(f"Client disconnected: {client_id}")
        self.client_disconnected.emit(client_id)
        try:
            client_socket.close()
        except:
            pass

    def broadcast_json(self, obj):
        # Broadcast JSON message, append newline for client line-based parsing
        msg = (json.dumps(obj) + "\n").encode()
        for cid, sock in list(self.clients.items()):
            try:
                sock.sendall(msg)
            except Exception as e:
                self.log.emit(f"Failed to send to {cid}: {e}")

############################################
# Playback Thread (for loop playback and stoppable by GUI)
############################################
class PlaybackThread(QThread):
    finished = pyqtSignal()
    log = pyqtSignal(str)

    def __init__(self, wav_path):
        super().__init__()
        self.wav_path = wav_path
        self._running = True

    def stop(self):
        # Main thread calls this method to request thread stop
        self._running = False

    def run(self):
        self.log.emit(f"Playback thread: looping {self.wav_path}")
        try:
            p = pyaudio.PyAudio()
            wf = wave.open(self.wav_path, 'rb')
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
            )
            # Loop playback until stop() is called
            while self._running:
                wf.rewind()
                data = wf.readframes(1024)
                while data and self._running:
                    stream.write(data)
                    data = wf.readframes(1024)
            try:
                stream.stop_stream()
                stream.close()
            except:
                pass
            try:
                wf.close()
            except:
                pass
            try:
                p.terminate()
            except:
                pass
            self.log.emit("Playback thread stopped")
        except Exception as e:
            self.log.emit(f"Playback error: {e}")
        self.finished.emit()

############################################
# WAV Generation Function (Utility)
############################################
def generate_ultrasonic_wav(filename, freq=19000, duration=0.1, fs=48000):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = 0.8 * np.sin(2 * np.pi * freq * t)
    data = (signal * 32767).astype(np.int16)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(data.tobytes())

############################################
# GUI Main Interface
############################################
class ServerGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Ultrasonic Server Controller")
        self.resize(900, 640)

        main_layout = QVBoxLayout()

        # Display local LAN IP and port at the top
        self.server_ip = get_local_ip()
        self.server_port = 5001
        self.ip_label = QLabel(f"Server IP: {self.server_ip}:{self.server_port}")
        main_layout.addWidget(self.ip_label)

        # Client List
        self.client_table = QTableWidget(0, 3)
        self.client_table.setHorizontalHeaderLabels(["Client ID", "Status", "Last File"])
        main_layout.addWidget(self.client_table)

        # Control Buttons and Parameter Row
        btn_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Recording (clients)")
        self.start_btn.clicked.connect(self.on_start_recording)
        btn_layout.addWidget(self.start_btn)

        self.play_btn = QPushButton("Start Ultrasonic (loop)")
        self.play_btn.clicked.connect(self.on_play)
        btn_layout.addWidget(self.play_btn)

        self.stop_play_btn = QPushButton("Stop Ultrasonic")
        self.stop_play_btn.clicked.connect(self.on_stop_play)
        self.stop_play_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_play_btn)

        self.stop_btn = QPushButton("Stop Recording (clients)")
        self.stop_btn.clicked.connect(self.on_stop_recording)
        btn_layout.addWidget(self.stop_btn)

        # WAV File Selection
        self.wav_path_edit = QLineEdit("")
        self.wav_path_edit.setPlaceholderText("Select or generate a wav file to play")
        btn_layout.addWidget(self.wav_path_edit)
        self.browse_wav_btn = QPushButton("Browse")
        self.browse_wav_btn.clicked.connect(self.on_browse_wav)
        btn_layout.addWidget(self.browse_wav_btn)

        # Generate WAV
        self.gen_btn = QPushButton("Generate Ultrasonic WAV")
        self.gen_btn.clicked.connect(self.on_generate_wav)
        btn_layout.addWidget(self.gen_btn)

        main_layout.addLayout(btn_layout)

        # File Naming Prefix Input
        prefix_layout = QHBoxLayout()
        prefix_layout.addWidget(QLabel("File prefix (e.g. user2-1-1):"))
        self.prefix_edit = QLineEdit("")
        prefix_layout.addWidget(self.prefix_edit)
        main_layout.addLayout(prefix_layout)

        # Log Box
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        main_layout.addWidget(self.log_box)

        self.setLayout(main_layout)

        # Start Server Thread (Bind to LAN IP)
        self.server_thread = ServerThread(host=self.server_ip, port=self.server_port)
        self.server_thread.log.connect(self.append_log)
        self.server_thread.client_connected.connect(self.add_client)
        self.server_thread.client_disconnected.connect(self.remove_client)
        self.server_thread.file_received.connect(self.on_file_received)
        self.server_thread.start()

        # Playback Thread Placeholder
        self.play_thread = None

    #################################
    # GUI Helper Functions
    #################################
    def append_log(self, text):
        self.log_box.append(text)

    def add_client(self, client_id):
        row = self.client_table.rowCount()
        self.client_table.insertRow(row)
        self.client_table.setItem(row, 0, QTableWidgetItem(client_id))
        self.client_table.setItem(row, 1, QTableWidgetItem("Connected"))
        self.client_table.setItem(row, 2, QTableWidgetItem(""))

    def remove_client(self, client_id):
        for row in range(self.client_table.rowCount()):
            if self.client_table.item(row, 0).text() == client_id:
                self.client_table.removeRow(row)
                break

    def update_last_file(self, client_id, filename):
        for row in range(self.client_table.rowCount()):
            if self.client_table.item(row, 0).text() == client_id:
                self.client_table.setItem(row, 2, QTableWidgetItem(filename))
                break

    #################################
    # Control Operations
    #################################
    def on_stop_play(self):
        # User clicks stop loop playback
        if hasattr(self, 'play_thread') and self.play_thread:
            try:
                # Set flag and call stop, if stop() method is implemented
                if hasattr(self.play_thread, 'stop'):
                    self.play_thread.stop()
                else:
                    self.play_thread._running = False
                self.append_log('Stop Ultrasonic requested')
            except Exception as e:
                self.append_log(f'Stop play error: {e}')

    def on_start_recording(self):
        # Update the prefix saved by the server
        self.server_thread.filename_prefix = self.prefix_edit.text().strip()

        if not self.server_thread.clients:
            self.append_log("No clients connected. Cannot start recording.")
            return

        cmd = {"cmd": "START_RECORD"}
        self.server_thread.broadcast_json(cmd)
        self.append_log("Sent START_RECORD to all clients")

    def on_play(self):
        wav_path = self.wav_path_edit.text().strip()
        if not wav_path or not os.path.exists(wav_path):
            self.append_log("WAV file not found. Please select or generate a WAV file first.")
            return

        # Playback started: disable some UI, but allow recording control
        self.play_btn.setEnabled(False)
        self.stop_play_btn.setEnabled(True)
        self.browse_wav_btn.setEnabled(False)
        self.gen_btn.setEnabled(False)

        # Create playback thread and loop playback
        self.play_thread = PlaybackThread(wav_path)
        # Default _running flag is True
        self.play_thread._running = True
        self.play_thread.log.connect(self.append_log)
        self.play_thread.finished.connect(self.on_play_finished)
        self.play_thread.start()

    def on_play_finished(self):
        # Triggered when playback thread finishes naturally or is stopped by stop()
        self.append_log("Playback stopped (GUI re-enabled)")
        self.start_btn.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.stop_play_btn.setEnabled(False)
        self.browse_wav_btn.setEnabled(True)
        self.gen_btn.setEnabled(True)
        self.play_thread = None

    def on_stop_recording(self):
        cmd = {"cmd": "STOP_RECORD"}
        self.server_thread.broadcast_json(cmd)
        self.append_log("Sent STOP_RECORD to all clients")

    #################################
    # WAV Helpers
    #################################
    def on_browse_wav(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select WAV file", "", "WAV files (*.wav)")
        if path:
            self.wav_path_edit.setText(path)

    def on_generate_wav(self):
        dialog = WavParamDialog(self)
        if dialog.exec():  # User clicks OK
            freq, duration, fs = dialog.get_values()
            out = os.path.abspath("ultrasonic.wav")
            try:
                generate_ultrasonic_wav(out, freq=freq, duration=duration, fs=fs)
                self.wav_path_edit.setText(out)
                self.append_log(f"Generated ultrasonic wav: {out} (freq={freq}Hz, duration={duration}s, fs={fs}Hz)")
            except Exception as e:
                self.append_log(f"Generate wav failed: {e}")

        # # Generate a default file ultrasonic.wav in the current directory
        # out = os.path.abspath("ultrasonic.wav")
        # try:
        #     generate_ultrasonic_wav(out)
        #     self.wav_path_edit.setText(out)
        #     self.append_log(f"Generated ultrasonic wav: {out}")
        # except Exception as e:
        #     self.append_log(f"Generate wav failed: {e}")

    #################################
    # File Received Callback
    #################################
    def on_file_received(self, client_id, saved_path):
        # Update table to show last filename
        self.append_log(f"File received from {client_id}: {saved_path}")
        self.update_last_file(client_id, os.path.basename(saved_path))

############################################
# Entry Point
############################################
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    win = ServerGUI()
    win.show()
    sys.exit(app.exec())

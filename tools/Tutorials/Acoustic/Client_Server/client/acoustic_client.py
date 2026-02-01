import socket
import json
import sys
import threading
import sounddevice as sd
import numpy as np
import wave
import time
import os

# ===================== CONFIG =====================
def resource_path(relative_path):
    # After packing with PyInstaller, resources are in the _MEIPASS temporary directory
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


DEFALUT_CONFIG_PATH = resource_path("config.json")
EXTERNAL_CONFIG_PATH = os.path.join(os.path.abspath("."), "config.json")

CONFIG_PATH = EXTERNAL_CONFIG_PATH if os.path.exists(EXTERNAL_CONFIG_PATH) else DEFALUT_CONFIG_PATH

def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError("config.json not found!")

    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

cfg = load_config()

SERVER_IP = cfg["server_ip"]
SERVER_PORT = cfg["server_port"]
CHANNELS = cfg["channels"]
DTYPE = cfg["d_type"]
SAMPLE_RATE = cfg["sample_rate"]
CLIENT_NAME = cfg["client_name"]

# Recording state and buffer
recording_flag = False
recorded_frames = []
frames_lock = threading.Lock()

# ============= Audio Recording Thread =============
def record_audio_thread():
    global recording_flag, recorded_frames
    with frames_lock:
        recorded_frames = []
    print("[CLIENT] Recording started...")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, blocksize=1024) as stream:
            while recording_flag:
                data, overflowed = stream.read(1024)
                # data shape: (1024, channels)
                with frames_lock:
                    recorded_frames.append(data.copy())
    except Exception as e:
        print(f"[CLIENT] Recording error: {e}")
    print("[CLIENT] Recording stopped.")

# ============= Save WAV =============
def save_wav(temp_wav_path="temp_record.wav"):
    global recorded_frames
    with frames_lock:
        if not recorded_frames:
            print("[CLIENT] No audio recorded.")
            return None
        audio_data = np.concatenate(recorded_frames, axis=0)

    # convert float32 (-1..1) to int16 safely (clip)
    if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
        audio_data = audio_data.astype(np.float32)
    audio_data = np.clip(audio_data, -1.0, 1.0)
    audio_int16 = (audio_data * 32767).astype(np.int16)

    try:
        with wave.open(temp_wav_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # int16
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        print(f"[CLIENT] WAV saved: {temp_wav_path}")
        return temp_wav_path
    except Exception as e:
        print(f"[CLIENT] Save wav error: {e}")
        return None

# ============= Upload File to Server =============
def upload_file(sock, filepath):
    try:
        filesize = os.path.getsize(filepath)
        filename = os.path.basename(filepath)
    except Exception as e:
        print(f"[CLIENT] File access error: {e}")
        return

    header = f"FILE_UPLOAD {filename} {filesize}\n"
    try:
        # Send header first
        sock.sendall(header.encode())
        # Wait a short while to reduce packet sticking between header and file (Server uses simple header-then-binary protocol)
        time.sleep(0.08)

        # Then send file binary data
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                sock.sendall(chunk)
        print("[CLIENT] File upload completed.")
    except Exception as e:
        print(f"[CLIENT] Upload error: {e}")

# ============= Main Receive Thread (Listen for Server Commands) =============
def listen_server(sock):
    global recording_flag

    buffer = b""
    while True:
        try:
            data = sock.recv(4096)
            if not data:
                print("[CLIENT] Server closed connection.")
                break
            buffer += data
        except Exception as e:
            print(f"[CLIENT] Socket recv error: {e}")
            break

        # When sending JSON, server code appends '\n' after each JSON object
        # So we split messages by newline (also compatible with single plain text commands)
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if not line:
                continue
            try:
                text = line.decode().strip()
            except:
                continue

            # Try parsing JSON first
            handled = False
            try:
                obj = json.loads(text)
                if isinstance(obj, dict) and "cmd" in obj:
                    cmd = obj["cmd"]
                    # There might be other fields, e.g., duration
                    if cmd == "START_RECORD":
                        if not recording_flag:
                            recording_flag = True
                            threading.Thread(target=record_audio_thread, daemon=True).start()
                            print("[CLIENT] START_RECORD (via JSON)")
                    elif cmd == "STOP_RECORD":
                        if recording_flag:
                            recording_flag = False
                            # give record thread a moment to flush
                            time.sleep(0.12)
                            wav_path = save_wav()
                            if wav_path:
                                upload_file(sock, wav_path)
                            print("[CLIENT] STOP_RECORD handled (via JSON)")
                    else:
                        print(f"[CLIENT] Unknown JSON cmd: {cmd}")
                    handled = True
            except json.JSONDecodeError:
                handled = False

            if handled:
                continue

            # If not JSON, handle as plain text protocol (compatible with old implementation)
            txt = text
            if txt.startswith("START_RECORD"):
                if not recording_flag:
                    recording_flag = True
                    threading.Thread(target=record_audio_thread, daemon=True).start()
                    print("[CLIENT] START_RECORD (plain)")
            elif txt.startswith("STOP_RECORD"):
                if recording_flag:
                    recording_flag = False
                    time.sleep(0.12)
                    wav_path = save_wav()
                    if wav_path:
                        upload_file(sock, wav_path)
                    print("[CLIENT] STOP_RECORD (plain)")
            else:
                print(f"[CLIENT] Unrecognized command: {txt}")

# ===================== Main Function =====================
def main():
    # Simple auto-reconnect logic
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((SERVER_IP, SERVER_PORT))
            print(f"[CLIENT] Connected to server {SERVER_IP}:{SERVER_PORT}")

            # ========== NEW: Immediately report client_name to server after connection success ==========
            hello = {"client_name": CLIENT_NAME}
            sock.sendall((json.dumps(hello) + "\n").encode())
            print(f"[CLIENT] Sent client_name: {CLIENT_NAME}")

            break
        except Exception as e:
            print(f"[CLIENT] Connect failed: {e}. Retrying in 2s...")
            time.sleep(2)

    threading.Thread(target=listen_server, args=(sock,), daemon=True).start()

    # Main thread keeps running (can be replaced with more complex control/heartbeat)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[CLIENT] Exiting.")
        try:
            sock.close()
        except:
            pass

if __name__ == "__main__":
    main()
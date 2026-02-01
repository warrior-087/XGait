# Acoustic Data Collection Tutorial

This directory contains tools for collecting acoustic data (CW) using standard microphones and speakers.

## System Architecture

### Python Client-Server Architecture (`Client_Server/`)

A distributed recording system implemented in Python:

| Component | Role | Description |
|-----------|------|-------------|
| **Server** (`server/`) | Control Node | Validates system setup and manages data storage |
| **Client** (`client/`) | Recording Node | Runs on the device connected to microphones for audio capture and transmission |

#### Core Files

- `acoustic_server.py` - Server main program
- `acoustic_client.py` - Client main program
- `ultrasonic.wav` - Base ultrasonic signal file

## Usage Guide

### Prerequisites

   Configure `config.json` in the client directory

### Startup Steps

1. **Start the Server (Speaker/Transmitter)**
   - Run on the computer connected to the **speaker**:
   ```bash
   python server/acoustic_server.py
   ```

2. **Start the Client (Microphone/Receiver)**
   - Run on the device connected to the **microphone**:
   ```bash
   python client/acoustic_client.py
   ```


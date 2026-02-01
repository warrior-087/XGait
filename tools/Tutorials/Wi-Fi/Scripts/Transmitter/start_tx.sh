#!/bin/bash

# ================= Configuration Area =================
# Tool path
TOOL_DIR="/home/minipc/linux-80211n-csitool-supplementary/injection"
# Packet injection parameters
NUMBER="100000000000000" # Number of packets (close to infinite)
LENGTH="100"             # Packet length
MODE="1"                 # Mode
DELAY="1000"             # Packet interval (us)
# ===========================================

if [ "$EUID" -ne 0 ]; then echo "Please run with sudo"; exit; fi

if [ -d "$TOOL_DIR" ]; then
    cd "$TOOL_DIR"
    echo "[TX Run] Start sending data packets..."
    echo "Params: Interval ${DELAY}us | Length ${LENGTH}"
    ./random_packets $NUMBER $LENGTH $MODE $DELAY
else
    echo "Error: Path not found $TOOL_DIR"
fi

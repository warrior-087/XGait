#!/bin/bash

# ================= Configuration Area =================
# Tool path (please modify to actual path)
TOOL_DIR="/home/minipc/linux-80211n-csitool-supplementary/netlink"
# Output filename (default with timestamp)
OUTPUT_FILE="log_$(date +%Y%m%d_%H%M%S).dat"
# ===========================================

if [ "$EUID" -ne 0 ]; then echo "Please run with sudo"; exit; fi

if [ -d "$TOOL_DIR" ]; then
    cd "$TOOL_DIR"
    echo "[RX Run] Start recording data to: $OUTPUT_FILE"
    echo "[RX Run] Press Ctrl+C to stop recording"
    ./log_to_file "$OUTPUT_FILE"
else
    echo "Error: Path not found $TOOL_DIR"
fi

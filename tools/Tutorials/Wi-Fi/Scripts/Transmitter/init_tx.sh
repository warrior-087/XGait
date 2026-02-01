#!/bin/bash

# ================= Configuration Area =================
IFACE="wlp3s0"         # Actual physical interface name
MON_IFACE="mon0"       # Monitor interface name

# Default parameters
DEFAULT_CHANNEL="64"   # Default channel
DEFAULT_BANDWIDTH="HT20"  # Default bandwidth mode
# ===========================================

# Argument parsing
CHANNEL=${1:-$DEFAULT_CHANNEL}
BANDWIDTH=${2:-$DEFAULT_BANDWIDTH}

# Show usage instructions
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: sudo $0 [Channel] [Bandwidth]"
    echo "Example: sudo $0 64 HT20"
    echo "         sudo $0 36 HT40+"
    echo ""
    echo "Arguments:"
    echo "  Channel: WiFi Channel (Default: $DEFAULT_CHANNEL)"
    echo "  Bandwidth: HT20, HT40+, HT40- (Default: $DEFAULT_BANDWIDTH)"
    exit 0
fi

if [ "$EUID" -ne 0 ]; then echo "Please run with sudo"; exit; fi

echo "[TX Init] Loading transmitter driver..."

# 1. Unload and reload driver (TX mode)
modprobe -r iwlwifi
modprobe iwlwifi csitool=0x1 debug=0x80000
sleep 2

# 2. Configure Monitor mode
if ip link show $MON_IFACE > /dev/null 2>&1; then
    iw dev $MON_IFACE del
fi

iw dev $IFACE interface add $MON_IFACE type monitor
iw $IFACE del

# 3. Activate interface and set channel
rfkill unblock wifi
ifconfig $MON_IFACE up
iw $MON_IFACE set channel $CHANNEL $BANDWIDTH

# 4. Lock transmission rate (TX specific step)
RATE_FILE=$(find /sys -name monitor_tx_rate)
if [ -n "$RATE_FILE" ]; then
    # Here sh -c is used to handle redirection permissions after sudo
    sh -c "echo 0x1c111 > $RATE_FILE"
    echo "[TX Init] Transmission rate locked (0x1c111)"
else
    echo "[Warning] monitor_tx_rate file not found, rate lock failed!"
fi

echo "[TX Init] Initialization complete! Channel: $CHANNEL $BANDWIDTH"

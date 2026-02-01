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

# Permission check
if [ "$EUID" -ne 0 ]; then echo "Please run with sudo"; exit; fi

echo "[RX Init] Loading receiver driver..."

# 1. Unload and reload driver (RX mode)
modprobe -r iwlwifi
modprobe iwlwifi csitool=0x1 connector_log=0x1
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

echo "[RX Init] 初始化完成！信道: $CHANNEL $BANDWIDTH"

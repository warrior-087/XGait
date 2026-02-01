# WiFi CSI Data Collection Tutorial

This directory contains the necessary software and scripts to configure MiniPCs for CSI collection using the Intel 5300 NIC.

## Prerequisites

| Requirement | Specification |
|-------------|----------------|
| **OS** | Ubuntu 14.04 LTS (Recommended for CSITool compatibility with Kernel 3.x/4.x) |
| **Hardware** | MiniPC with Intel 5300 NIC |
| **Driver** | `iwlwifi` properly installed with CSI logging enabled |

## Directory Structure

```text
Wi-Fi/
├── Configuration/
│   └── minipc_setup_guide.md    # Detailed setup guide for MiniPCs
└── Scripts/
    ├── Receiver/                # Scripts to initialize Monitor Mode and log CSI
    │   ├── init_rx.sh           # Sets up monitor mode on interface wlan0
    │   └── start_rx.sh          # Starts the logging process
    └── Transmitter/             # Scripts to initialize Injection Mode and start sending packets
        ├── init_tx.sh           # Configures the injection interface
        └── start_tx.sh          # Starts sending packets at a fixed rate
```

## Quick Start

### Receiver Setup (on 3 Rx MiniPCs)

```bash
cd Scripts/Receiver
sudo ./init_rx.sh
sudo ./start_rx.sh
```

### Transmitter Setup (on 1 Tx MiniPC)

```bash
cd Scripts/Transmitter
sudo ./init_tx.sh
sudo ./start_tx.sh
```

## Dependencies

Since this repository does not include third-party binaries, you must install the following:

1. `linux-80211n-csitool-supplementary`
2. `lorcon-old` (for packet injection)

Please refer to [minipc_setup_guide.md](Configuration/minipc_setup_guide.md) for detailed installation instructions.

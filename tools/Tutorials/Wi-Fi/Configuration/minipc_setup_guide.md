# MiniPC Setup & Configuration Guide

This guide details the steps to configure the MiniPCs (running Ubuntu 20.04 LTS) for WiFi CSI data collection.

## 1. System Configuration

### 1.1 Network Setup (Static IP & Autoconnect)
Ensure the MiniPC automatically connects to the wired network on boot.

```bash
# Release current IP (Note: Replace 'enp2s0' with your actual interface name)
sudo dhclient -r enp2s0
# Renew IPv4 address
sudo dhclient -4 enp2s0

# Identify connection name
nmcli connection show

# Configure auto-connection (Replace "Wired connection 1" with your actual connection name)
sudo nmcli connection modify "Wired connection 1" connection.autoconnect yes
sudo nmcli connection modify "Wired connection 1" ipv4.method auto
sudo systemctl restart NetworkManager
sudo nmcli connection up "Wired connection 1"
```

### 1.2 SSH Installation
To enable remote management of the MiniPCs:

```bash
sudo apt update
sudo apt install openssh-server
sudo systemctl start ssh
sudo systemctl enable ssh
```

## 2. Kernel Downgrade (Critical)
The Intel 5300 CSI Tool requires specific Linux kernel versions. For Ubuntu 20.04, we downgrade to Kernel 5.4.0-26.

```bash
# Install Kernel 5.4.0-26
sudo apt install linux-image-5.4.0-26-generic linux-headers-5.4.0-26-generic
sudo apt install linux-modules-5.4.0-26-generic linux-modules-extra-5.4.0-26-generic

# Update GRUB to boot from the new kernel by default
sudo vim /etc/default/grub
# Change the GRUB_DEFAULT line to:
# GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux 5.4.0-26-generic"

# Apply changes and reboot
sudo update-grub
sudo reboot
```

## 3. CSI Tool Installation
Since the original CSI Tool was designed for older kernels (Linux 3.x/4.x), porting it to Ubuntu 20.04 (Linux 5.4+) requires modifying the kernel driver.

### 3.1 Overview
The process involves three main parts (details omitted for brevity):
1.  **Kernel Source**: Using Linux 5.4.30.
2.  **Drivers**: Installing the patched `iwlwifi` drivers.
3.  **Userspace Tools**: Compiling the CSITool user-space utilities.

> **Note**: For the full porting and installation procedure, please contact the author.

## 4. Userspace Tools Installation

### 4.1 Linux 802.11n CSI Tool (Supplements)
Compile the `log_to_file` tool to capture CSI data.
```bash
sudo apt install git
git clone https://github.com/dhalperi/linux-80211n-csitool-supplementary.git

# Install Firmware
sudo cp linux-80211n-csitool-supplementary/firmware/iwlwifi-5000-2.ucode.sigcomm2010 /lib/firmware/

# Compile netlink tool
cd linux-80211n-csitool-supplementary/netlink/
make
```

### 4.2 Lorcon & Injection (For Transmitter)
Install the library required for packet injection.
```bash
# Install dependencies
sudo apt-get install libnl-genl-3-dev

# Install Lorcon
git clone https://github.com/dhalperi/lorcon-old.git
cd lorcon-old/
./configure
make
sudo make install

# Compile injection tool
cd ../linux-80211n-csitool-supplementary/injection/
make
```

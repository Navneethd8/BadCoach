#!/usr/bin/env bash
# Install proprietary NVIDIA drivers on a fresh GPU EC2 instance, then reboot.
#
# Supported: Ubuntu (ubuntu-drivers autoinstall). Debian: use non-free nvidia packages manually.
# For Amazon Linux, use an AWS GPU / Deep Learning AMI or follow NVIDIA AL install docs;
#   this script exits with instructions if ID=amzn.
#
# Usage (on the instance, after SSH):
#   chmod +x scripts/ec2/install_nvidia_driver_reboot.sh
#   sudo ./scripts/ec2/install_nvidia_driver_reboot.sh
#
# Env:
#   INSTALL_SKIP_REBOOT=1  — install only, do not reboot
#   INSTALL_SKIP_SLEEP=1   — reboot immediately (no 15s delay)

set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root: sudo $0" >&2
  exit 1
fi

if [[ -f /etc/os-release ]]; then
  # shellcheck source=/dev/null
  . /etc/os-release
else
  echo "Cannot read /etc/os-release" >&2
  exit 1
fi

install_ubuntu() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y ubuntu-drivers-common
  ubuntu-drivers autoinstall
}

case "${ID:-}" in
  ubuntu)
    install_ubuntu
    ;;
  debian)
    echo "Debian detected. Enable non-free and install an nvidia-driver package for your kernel," >&2
    echo "or switch to Ubuntu on the instance for ubuntu-drivers autoinstall." >&2
    exit 2
    ;;
  amzn)
    echo "Amazon Linux (${VERSION_ID:-}) detected." >&2
    echo "This script does not auto-install drivers on amzn." >&2
    echo "Options: launch an AWS Deep Learning AMI / GPU-optimized AMI with drivers preinstalled," >&2
    echo "or follow: https://docs.nvidia.com/cuda/cuda-installation-guide-linux-amazon-linux/" >&2
    exit 2
    ;;
  *)
    echo "Unsupported OS ID=${ID:-unknown}. Use Ubuntu, or install drivers manually." >&2
    exit 2
    ;;
esac

if [[ "${INSTALL_SKIP_REBOOT:-0}" == "1" ]]; then
  echo "INSTALL_SKIP_REBOOT=1: skipping reboot. Run nvidia-smi after a manual reboot."
  exit 0
fi

if [[ "${INSTALL_SKIP_SLEEP:-0}" != "1" ]]; then
  echo "Rebooting in 15 seconds (Ctrl-C to cancel, or INSTALL_SKIP_SLEEP=1 for immediate reboot)..."
  sleep 15
fi

reboot

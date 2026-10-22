#!/usr/bin/env bash
# Collect the "Hardware and system" rows of the llama.cpp vs TensorRT-LLM Shared Setup
# Agreement on this Jetson, so both boards can be compared row by row.
#
# Usage (on the Jetson host, not inside a container):
#   bash tensorrt/collect_board_info.sh [MODEL_DIR] > tensorrt/board_info_$(hostname).txt
# MODEL_DIR (optional) is where the engines/models load from, to report its storage device.

MODEL_DIR="${1:-$HOME/tensorrt-edgellm-workspace}"

section() { printf '\n== %s ==\n' "$1"; }
run() { printf '$ %s\n' "$*"; "$@" 2>&1 || true; }

section "Board"
run cat /proc/device-tree/model; echo
run uname -r

section "JetPack / L4T"
run cat /etc/nv_tegra_release
run dpkg-query -W -f='${Package} ${Version}\n' nvidia-jetpack nvidia-l4t-core

section "CUDA / cuDNN / TensorRT"
if command -v nvcc >/dev/null; then run nvcc --version; else run /usr/local/cuda/bin/nvcc --version; fi
run dpkg-query -W -f='${Package} ${Version}\n' 'libcudnn*' 'libnvinfer*' 'tensorrt*'

section "Power mode (agreement: 15W, locked)"
run sudo nvpmodel -q

section "Clocks and fan (agreement: jetson_clocks not forced, fan dynamic)"
run sudo jetson_clocks --show
for f in /sys/devices/platform/pwm-fan/hwmon/hwmon*/pwm1 /sys/class/hwmon/hwmon*/pwm1; do
  [ -r "$f" ] && echo "$f = $(cat "$f")"
done

section "Memory and swap"
run free -m
run swapon --show
run systemctl is-enabled nvzramconfig

section "Storage the models load from: $MODEL_DIR"
run df -h "$MODEL_DIR"
run findmnt -T "$MODEL_DIR"

section "Docker"
run docker ps

section "Other heavy processes (agreement: none during runs)"
run sh -c "ps -eo pid,pcpu,pmem,rss,comm --sort=-rss | head -15"

section "TensorRT Edge-LLM checkout"
EDGE_DIR="${EDGE_LLM_PATH:-$HOME/TensorRT-Edge-LLM}"
run git -C "$EDGE_DIR" describe --tags --always
run git -C "$EDGE_DIR" rev-parse HEAD

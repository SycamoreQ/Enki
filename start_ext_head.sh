#!/bin/bash
echo "Starting Ray Head Node (macOS - External GPU Access)"

# Kill existing Ray
ray stop || true

# Get your LAN IP
LAN_IP=$(ipconfig getifaddr en0 || ipconfig getifaddr en1 || ipconfig getifaddr enp0s3 || hostname -I | cut -d' ' -f1 || echo "192.168.1.10")
echo "Detected LAN IP: $LAN_IP"

# Set shared temp dir and macOS override
export RAY_TMPDIR="/tmp/ray_session_6379"
mkdir -p "$RAY_TMPDIR"
export RAY_ENABLE_MAC_LARGE_OBJECT_STORE=1

# Start Ray HEAD (single line - no backslashes needed)
ray start --head --port=6379 --node-ip-address="$LAN_IP" --dashboard-port=8265 --dashboard-host=0.0.0.0 --object-store-memory=2000000000 --num-gpus=0 --temp-dir="$RAY_TMPDIR" --disable-usage-stats

echo "Ray Head Started"
echo "  Connect URL: $LAN_IP:6379"
echo "  Dashboard: http://$LAN_IP:8265"
echo "  Temp dir: $RAY_TMPDIR"
echo "  GPU workers run:"
echo "    export RAY_TMPDIR=\"$RAY_TMPDIR\""
echo "    ray start --address=\"$LAN_IP:6379\" --num-gpus=1"

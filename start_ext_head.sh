echo "Starting Ray Head Node (macOS - External GPU Access)"

# Force kill ALL Ray processes and clean ports
pkill -f ray || true
sudo lsof -ti:6379,8265 | xargs sudo kill -9 || true
ray stop --force || true
sleep 2

# Get LAN IP
LAN_IP=$(ipconfig getifaddr en0 || ipconfig getifaddr en1 || ipconfig getifaddr enp0s3 || hostname -I | cut -d' ' -f1 || echo "192.168.1.10")
echo "Detected LAN IP: $LAN_IP"

# Use DIFFERENT dashboard port to avoid conflicts
ray start --head \
  --port=6379 \
  --node-ip-address="$LAN_IP" \
  --dashboard-port=8265 \
  --dashboard-host=0.0.0.0 \
  --object-spilling-directory="/tmp/ray_spill" \
  --object-store-memory=2000000000 \
  --num-gpus=0 \
  --disable-usage-stats

echo "Ray Head Started"
echo "  Connect URL: $LAN_IP:6379"
echo "  Dashboard: http://$LAN_IP:8266"
echo "  Temp dir: $RAY_TMPDIR"
echo "  Worker command:"
echo "    export RAY_TMPDIR=\"$RAY_TMPDIR\""
echo "    ray start --address=\"$LAN_IP:6379\" --num-gpus=1"

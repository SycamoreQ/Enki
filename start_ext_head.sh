echo "Starting Ray Head Node (macOS - External GPU Access)"

# Kill existing Ray

ray stop || true

# Get your LAN IP
LAN_IP=$(ipconfig getifaddr en0 || ipconfig getifaddr en1 || ipconfig getifaddr enp0s3 || hostname -I | cut -d' ' -f1 || echo "192.168.1.10")
echo "Detected LAN IP: $LAN_IP"

# Start Ray HEAD with EXTERNAL binding
ray start \
  --head \
  --port=6379 \
  --node-ip-address="$LAN_IP" \
  --dashboard-port=8265 \
  --dashboard-host=0.0.0.0 \
  --object-store-memory=8000000000 \
  --num-gpus=0 \
  --include-dashboard=true \
  --temp-dir=$HOME/ray_tmp \
  --disable-usage-stats

echo "Ray Head Started"
echo "  Connect URL: $LAN_IP:6379"
echo "  Dashboard: http://$LAN_IP:8265"
echo "  GPU workers run:"
echo "    ray start --address='$LAN_IP:6379' --num-gpus=1"

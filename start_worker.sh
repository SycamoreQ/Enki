#!/bin/bash
# Start Worker Node for Distributed Training

set -e

if [ -z "$1" ]; then
    echo "Usage: ./start_worker.sh <worker_name> [config_file]"
    echo ""
    echo "Example: ./start_worker.sh rl_worker_1"
    echo ""
    echo "Available workers:"
    python3 -c "
import yaml
try:
    config = yaml.safe_load(open('utils/config/cluster_config.yaml'))
    for w in config['rl_training_workers']:
        if w.get('host'):
            print(f\"  - {w['name']}\")
except Exception as e:
    print(f'  Error reading config: {e}')
"
    exit 1
fi

WORKER_NAME=$1
CONFIG_FILE="${2:-utils/config/cluster_config.yaml}"

echo "=================================================="
echo "  Starting Distributed Training Worker"
echo "=================================================="
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

# Extract master config
MASTER_HOST=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['master_node']['host'])")
RAY_PORT=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['master_node']['ray_port'])")
DASHBOARD_PORT=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['master_node']['dashboard_port'])")

# Get worker-specific config
WORKER_CPU=$(python3 -c "
import yaml
config = yaml.safe_load(open('$CONFIG_FILE'))
for w in config['rl_training_workers']:
    if w['name'] == '$WORKER_NAME':
        print(w['resources']['CPU'])
        break
else:
    print('NOT_FOUND')
")

if [ "$WORKER_CPU" == "NOT_FOUND" ]; then
    echo "❌ Worker '$WORKER_NAME' not found in config"
    exit 1
fi

WORKER_GPU=$(python3 -c "
import yaml
config = yaml.safe_load(open('$CONFIG_FILE'))
for w in config['rl_training_workers']:
    if w['name'] == '$WORKER_NAME':
        print(w['resources']['GPU'])
        break
")

WORKER_MEMORY=$(python3 -c "
import yaml
config = yaml.safe_load(open('$CONFIG_FILE'))
for w in config['rl_training_workers']:
    if w['name'] == '$WORKER_NAME':
        print(w['resources']['memory'])
        break
")

echo "Worker: $WORKER_NAME"
echo "Master: $MASTER_HOST:$RAY_PORT"
echo "Resources:"
echo "  CPUs: $WORKER_CPU"
echo "  GPUs: $WORKER_GPU"
echo "  Memory: $(python3 -c "print(f'{$WORKER_MEMORY/1e9:.1f}GB')")"
echo ""

# Check if Ray is already running
if ray status > /dev/null 2>&1; then
    echo "⚠️  Ray is already running on this machine"
    read -p "Stop and restart? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping Ray..."
        ray stop
        sleep 3
    else
        echo "❌ Exiting"
        exit 1
    fi
fi

# Check GPU
if [ "$WORKER_GPU" -gt 0 ]; then
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Status:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.free --format=csv,noheader
        echo ""
    else
        echo "⚠️  Warning: GPU requested but nvidia-smi not available"
        echo ""
    fi
fi

echo "Connecting to master at $MASTER_HOST:$RAY_PORT..."

ray start \
    --address="$MASTER_HOST:$RAY_PORT" \
    --num-cpus=$WORKER_CPU \
    --num-gpus=$WORKER_GPU \
    --memory=$WORKER_MEMORY \
    --resources='{"rl_trainer": 1}' \
    --disable-usage-stats

sleep 3

# Check if connected
if ! ray status > /dev/null 2>&1; then
    echo "❌ Failed to connect to Ray cluster"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check master is running: ssh user@$MASTER_HOST 'ray status'"
    echo "  2. Check network: ping $MASTER_HOST"
    echo "  3. Check port: telnet $MASTER_HOST $RAY_PORT"
    echo "  4. Check firewall: sudo ufw allow $RAY_PORT/tcp"
    exit 1
fi

echo ""
echo "✓ Worker connected to cluster"
echo ""

# Display cluster info
echo "Cluster Status:"
ray status
echo ""

echo "=================================================="
echo "  Worker Node Running: $WORKER_NAME"
echo "=================================================="
echo ""
echo "✓ Worker will process jobs assigned by coordinator"
echo "✓ Monitor: http://$MASTER_HOST:$DASHBOARD_PORT"
echo "✓ Stop: ray stop"
echo ""
echo "Worker is ready. Press Ctrl+C to stop monitoring."
echo ""

# Monitor Ray logs
tail -f /tmp/ray/session_latest/logs/worker*.out 2>/dev/null || sleep infinity
set -e

if [ -z "$1" ]; then
    echo "Usage: ./start_worker_improved.sh <worker_name> [config_file]"
    echo ""
    echo "Example: ./start_worker_improved.sh rl_worker_1"
    echo ""
    echo "Available workers:"
    python3 -c "
import yaml
config = yaml.safe_load(open('utils/config/cluster_config.yaml'))
for w in config['rl_training_workers']:
    if w['host']:
        print(f\"  - {w['name']}\")
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
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Extract config using Python
MASTER_HOST=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['master_node']['host'])")
RAY_PORT=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['master_node']['ray_port'])")

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
    echo "Worker '$WORKER_NAME' not found in config"
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
echo "  Memory: $(echo "scale=2; $WORKER_MEMORY / 1000000000" | bc)GB"
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
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
        echo ""
    else
        echo " Warning: GPU requested but nvidia-smi not available"
    fi
fi

echo "Connecting to master at $MASTER_HOST:$RAY_PORT..."

ray start \
    --address="$MASTER_HOST:$RAY_PORT" \
    --num-cpus=$WORKER_CPU \
    --num-gpus=$WORKER_GPU \
    --memory=$WORKER_MEMORY \
    --resources='{"rl_trainer": 1}'

# Check if connected
if ! ray status > /dev/null 2>&1; then
    echo "Failed to connect to Ray cluster"
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
echo "Worker will process jobs assigned by the coordinator"
echo "Monitor at: http://$MASTER_HOST:$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['master_node']['dashboard_port'])")"
echo ""
echo "Stop: ray stop"
echo "=================================================="
echo ""

# Keep script running to show logs
echo "Worker is ready. Press Ctrl+C to stop monitoring."
echo ""

# Monitor Ray logs
tail -f /tmp/ray/session_latest/logs/worker*.out 2>/dev/null || sleep infinity
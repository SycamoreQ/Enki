#!/bin/bash
# All-in-one cluster startup script
# Usage: ./start_cluster.sh master  OR  ./start_cluster.sh worker rl_worker_1

set -e

MODE="${1:-help}"
CONFIG_FILE="utils/config/cluster_config.yaml"

if [ "$MODE" = "help" ] || [ "$MODE" = "--help" ] || [ "$MODE" = "-h" ]; then
    echo "Usage:"
    echo "  ./start_cluster.sh master"
    echo "  ./start_cluster.sh worker <worker_name>"
    echo ""
    echo "Examples:"
    echo "  ./start_cluster.sh master"
    echo "  ./start_cluster.sh worker rl_worker_1"
    echo ""
    exit 0
fi

if [ "$MODE" = "master" ]; then
    exec ./start_master.sh "$CONFIG_FILE"
    
elif [ "$MODE" = "worker" ]; then
    WORKER_NAME="${2:-}"
    
    if [ -z "$WORKER_NAME" ]; then
        echo "❌ Worker name required"
        echo "Usage: ./start_cluster.sh worker <worker_name>"
        echo ""
        echo "Available workers:"
        python3 -c "
import yaml
config = yaml.safe_load(open('$CONFIG_FILE'))
for w in config['rl_training_workers']:
    if w.get('host'):
        print(f\"  - {w['name']}\")
"
        exit 1
    fi
    
    exec ./start_worker.sh "$WORKER_NAME" "$CONFIG_FILE"
    
else
    echo "❌ Unknown mode: $MODE"
    echo "Usage: ./start_cluster.sh {master|worker} [worker_name]"
    exit 1
fi
#!/bin/bash
# One-command cluster startup with automatic cleanup

set -e

MODE="${1:-master}"
CONFIG_FILE="${2:-utils/config/cluster_config.yaml}"

echo "=================================================="
echo "  Ray Distributed Training Cluster"
echo "=================================================="
echo ""

# Function to cleanup Ray
cleanup_ray() {
    echo "Cleaning up Ray..."
    ray stop 2>/dev/null || true
    sleep 1
    
    pkill -9 -f "ray::" 2>/dev/null || true
    pkill -9 -f "raylet" 2>/dev/null || true
    pkill -9 -f "gcs_server" 2>/dev/null || true
    pkill -9 -f "dashboard" 2>/dev/null || true
    
    sleep 1
    rm -rf /tmp/ray/* 2>/dev/null || true
    
    echo "✓ Cleanup complete"
}

# Function to extract config
get_config() {
    python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))$1)"
}

if [ "$MODE" = "master" ]; then
    echo "Mode: MASTER NODE"
    echo ""
    
    # Cleanup first
    cleanup_ray
    
    # Get config
    MASTER_HOST=$(get_config "['master_node']['host']")
    RAY_PORT=$(get_config "['master_node']['ray_port']")
    DASHBOARD_PORT=$(get_config "['master_node']['dashboard_port']")
    
    echo "Configuration:"
    echo "  Host: $MASTER_HOST"
    echo "  Ray Port: $RAY_PORT"
    echo "  Dashboard: $DASHBOARD_PORT"
    echo ""
    
    echo "Starting Ray head node..."
    ray start --head \
        --port=$RAY_PORT \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=$DASHBOARD_PORT \
        --num-cpus=2 \
        --num-gpus=0 \
        --include-dashboard=true \
        --disable-usage-stats
    
    sleep 2
    
    # Verify
    if ! ray status > /dev/null 2>&1; then
        echo "❌ Failed to start Ray"
        echo "Check logs: /tmp/ray/session_latest/logs/raylet.out"
        exit 1
    fi
    
    echo ""
    echo "✓ Ray head node started"
    echo "  Dashboard: http://$MASTER_HOST:$DASHBOARD_PORT"
    echo ""
    
    # Start coordinator - check for file in multiple locations
    echo "Starting training coordinator..."
    
    if [ -f "distributed_master.py" ]; then
        python3 distributed_master.py --config "$CONFIG_FILE" &
    elif [ -f "distribute/workers/distributed_master.py" ]; then
        python3 distribute/workers/distributed_master.py --config "$CONFIG_FILE" &
    else
        echo "❌ distributed_master.py not found"
        echo "Expected locations:"
        echo "  - distributed_master.py"
        echo "  - distribute/workers/distributed_master.py"
        ray stop
        exit 1
    fi
    
    COORDINATOR_PID=$!
    sleep 2
    
    # Check if coordinator started
    if ! ps -p $COORDINATOR_PID > /dev/null; then
        echo "❌ Coordinator failed to start"
        exit 1
    fi
    
    echo "✓ Coordinator started (PID: $COORDINATOR_PID)"
    echo ""
    echo "=================================================="
    echo "  Master Node Running"
    echo "=================================================="
    echo ""
    echo "Ray Status: ray status"
    echo "Dashboard: http://$MASTER_HOST:$DASHBOARD_PORT"
    echo "Stop: ray stop"
    echo ""
    echo "Press Ctrl+C to stop"
    echo ""
    
    # Keep running
    trap cleanup_ray EXIT
    wait $COORDINATOR_PID

elif [ "$MODE" = "worker" ]; then
    WORKER_NAME="${3:-}"
    
    if [ -z "$WORKER_NAME" ]; then
        echo "Usage: $0 worker [config_file] <worker_name>"
        echo ""
        echo "Available workers:"
        python3 -c "
import yaml
config = yaml.safe_load(open('$CONFIG_FILE'))
for w in config['rl_training_workers']:
    if w['host']:
        print(f\"  - {w['name']}\")
"
        exit 1
    fi
    
    echo "Mode: WORKER NODE ($WORKER_NAME)"
    echo ""
    
    # Cleanup first
    cleanup_ray
    
    # Get config
    MASTER_HOST=$(get_config "['master_node']['host']")
    RAY_PORT=$(get_config "['master_node']['ray_port']")
    
    WORKER_CPU=$(python3 -c "
import yaml
config = yaml.safe_load(open('$CONFIG_FILE'))
for w in config['rl_training_workers']:
    if w['name'] == '$WORKER_NAME':
        print(w['resources']['CPU'])
        break
else:
    print('0')
")
    
    if [ "$WORKER_CPU" = "0" ]; then
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
    
    echo "Configuration:"
    echo "  Master: $MASTER_HOST:$RAY_PORT"
    echo "  CPUs: $WORKER_CPU"
    echo "  GPUs: $WORKER_GPU"
    echo "  Memory: $(python3 -c "print(f'{$WORKER_MEMORY/1e9:.1f}GB')")"
    echo ""
    
    # Check GPU if needed
    if [ "$WORKER_GPU" -gt 0 ]; then
        if command -v nvidia-smi &> /dev/null; then
            echo "GPU Status:"
            nvidia-smi --query-gpu=index,name,utilization.gpu,memory.free --format=csv,noheader
            echo ""
        fi
    fi
    
    echo "Connecting to master..."
    ray start \
        --address="$MASTER_HOST:$RAY_PORT" \
        --num-cpus=$WORKER_CPU \
        --num-gpus=$WORKER_GPU \
        --memory=$WORKER_MEMORY \
        --resources='{"rl_trainer": 1}' \
        --disable-usage-stats
    
    sleep 2
    
    if ! ray status > /dev/null 2>&1; then
        echo "❌ Failed to connect to cluster"
        exit 1
    fi
    
    echo ""
    echo "✓ Worker connected"
    echo ""
    echo "Cluster Status:"
    ray status
    echo ""
    echo "=================================================="
    echo "  Worker Running: $WORKER_NAME"
    echo "=================================================="
    echo ""
    echo "Stop: ray stop"
    echo ""
    echo "Press Ctrl+C to stop"
    echo ""
    
    # Keep running
    trap cleanup_ray EXIT
    tail -f /tmp/ray/session_latest/logs/worker*.out 2>/dev/null || sleep infinity

else
    echo "Usage: $0 {master|worker} [config_file] [worker_name]"
    echo ""
    echo "Examples:"
    echo "  $0 master                           # Start master node"
    echo "  $0 worker utils/config/cluster_config.yaml rl_worker_1"
    echo ""
    exit 1
fi
#!/bin/bash
# One-command cluster startup with automatic cleanup

set -e

MODE="${1:-master}"
CONFIG_FILE="${2:-utils/config/cluster_config.yaml}"

echo "=================================================="
echo "  Ray Distributed Training Cluster"
echo "=================================================="
echo ""

# Function to cleanup Ray
cleanup_ray() {
    echo "Cleaning up Ray..."
    ray stop 2>/dev/null || true
    sleep 1
    
    pkill -9 -f "ray::" 2>/dev/null || true
    pkill -9 -f "raylet" 2>/dev/null || true
    pkill -9 -f "gcs_server" 2>/dev/null || true
    pkill -9 -f "dashboard" 2>/dev/null || true
    
    sleep 1
    rm -rf /tmp/ray/* 2>/dev/null || true
    
    echo "✓ Cleanup complete"
}

# Function to extract config
get_config() {
    python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))$1)"
}

if [ "$MODE" = "master" ]; then
    echo "Mode: MASTER NODE"
    echo ""
    
    # Cleanup first
    cleanup_ray
    
    # Get config
    MASTER_HOST=$(get_config "['master_node']['host']")
    RAY_PORT=$(get_config "['master_node']['ray_port']")
    DASHBOARD_PORT=$(get_config "['master_node']['dashboard_port']")
    
    echo "Configuration:"
    echo "  Host: $MASTER_HOST"
    echo "  Ray Port: $RAY_PORT"
    echo "  Dashboard: $DASHBOARD_PORT"
    echo ""
    
    echo "Starting Ray head node..."
    ray start --head \
        --port=$RAY_PORT \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=$DASHBOARD_PORT \
        --num-cpus=2 \
        --num-gpus=0 \
        --include-dashboard=true \
        --disable-usage-stats
    
    sleep 2
    
    # Verify
    if ! ray status > /dev/null 2>&1; then
        echo "❌ Failed to start Ray"
        echo "Check logs: /tmp/ray/session_latest/logs/raylet.out"
        exit 1
    fi
    
    echo ""
    echo "✓ Ray head node started"
    echo "  Dashboard: http://$MASTER_HOST:$DASHBOARD_PORT"
    echo ""
    
    # Start coordinator
    echo "Starting training coordinator..."
    python3 distributed_master.py --config "$CONFIG_FILE" &
    
    COORDINATOR_PID=$!
    sleep 2
    
    # Check if coordinator started
    if ! ps -p $COORDINATOR_PID > /dev/null; then
        echo "❌ Coordinator failed to start"
        exit 1
    fi
    
    echo "✓ Coordinator started (PID: $COORDINATOR_PID)"
    echo ""
    echo "=================================================="
    echo "  Master Node Running"
    echo "=================================================="
    echo ""
    echo "Ray Status: ray status"
    echo "Dashboard: http://$MASTER_HOST:$DASHBOARD_PORT"
    echo "Stop: ./cleanup_ray.sh"
    echo ""
    echo "Press Ctrl+C to stop"
    echo ""
    
    # Keep running
    trap cleanup_ray EXIT
    wait $COORDINATOR_PID

elif [ "$MODE" = "worker" ]; then
    WORKER_NAME="${3:-}"
    
    if [ -z "$WORKER_NAME" ]; then
        echo "Usage: $0 worker [config_file] <worker_name>"
        echo ""
        echo "Available workers:"
        python3 -c "
import yaml
config = yaml.safe_load(open('$CONFIG_FILE'))
for w in config['rl_training_workers']:
    if w['host']:
        print(f\"  - {w['name']}\")
"
        exit 1
    fi
    
    echo "Mode: WORKER NODE ($WORKER_NAME)"
    echo ""
    
    # Cleanup first
    cleanup_ray
    
    # Get config
    MASTER_HOST=$(get_config "['master_node']['host']")
    RAY_PORT=$(get_config "['master_node']['ray_port']")
    
    WORKER_CPU=$(python3 -c "
import yaml
config = yaml.safe_load(open('$CONFIG_FILE'))
for w in config['rl_training_workers']:
    if w['name'] == '$WORKER_NAME':
        print(w['resources']['CPU'])
        break
else:
    print('0')
")
    
    if [ "$WORKER_CPU" = "0" ]; then
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
    
    echo "Configuration:"
    echo "  Master: $MASTER_HOST:$RAY_PORT"
    echo "  CPUs: $WORKER_CPU"
    echo "  GPUs: $WORKER_GPU"
    echo "  Memory: $(python3 -c "print(f'{$WORKER_MEMORY/1e9:.1f}GB')")"
    echo ""
    
    # Check GPU if needed
    if [ "$WORKER_GPU" -gt 0 ]; then
        if command -v nvidia-smi &> /dev/null; then
            echo "GPU Status:"
            nvidia-smi --query-gpu=index,name,utilization.gpu,memory.free --format=csv,noheader
            echo ""
        fi
    fi
    
    echo "Connecting to master..."
    ray start \
        --address="$MASTER_HOST:$RAY_PORT" \
        --num-cpus=$WORKER_CPU \
        --num-gpus=$WORKER_GPU \
        --memory=$WORKER_MEMORY \
        --resources='{"rl_trainer": 1}' \
        --disable-usage-stats
    
    sleep 2
    
    if ! ray status > /dev/null 2>&1; then
        echo "❌ Failed to connect to cluster"
        exit 1
    fi
    
    echo ""
    echo "✓ Worker connected"
    echo ""
    echo "Cluster Status:"
    ray status
    echo ""
    echo "=================================================="
    echo "  Worker Running: $WORKER_NAME"
    echo "=================================================="
    echo ""
    echo "Stop: ./cleanup_ray.sh"
    echo ""
    echo "Press Ctrl+C to stop"
    echo ""
    
    # Keep running
    trap cleanup_ray EXIT
    tail -f /tmp/ray/session_latest/logs/worker*.out 2>/dev/null || sleep infinity

else
    echo "Usage: $0 {master|worker} [config_file] [worker_name]"
    echo ""
    echo "Examples:"
    echo "  $0 master                           # Start master node"
    echo "  $0 worker utils/config/cluster_config.yaml rl_worker_1"
    echo ""
    exit 1
fi
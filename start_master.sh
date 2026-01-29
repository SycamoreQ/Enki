set -e

CONFIG_FILE="${1:-utils/config/cluster_config.yaml}"

echo "=================================================="
echo "  Starting Distributed Training Master Node"
echo "=================================================="
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

# Extract config values using Python
MASTER_HOST=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['master_node']['host'])")
RAY_PORT=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['master_node']['ray_port'])")
DASHBOARD_PORT=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['master_node']['dashboard_port'])")

echo "Configuration:"
echo "  Host: $MASTER_HOST"
echo "  Ray Port: $RAY_PORT"
echo "  Dashboard: $DASHBOARD_PORT"
echo ""

# Check if Ray is already running
if ray status > /dev/null 2>&1; then
    echo "⚠️  Ray is already running"
    read -p "Clean up and restart? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Cleaning up..."
        ray stop
        sleep 2
        pkill -9 -f "ray::" 2>/dev/null || true
        pkill -9 -f "raylet" 2>/dev/null || true
        pkill -9 -f "gcs_server" 2>/dev/null || true
        sleep 1
        rm -rf /tmp/ray/* 2>/dev/null || true
    else
        echo "❌ Exiting"
        exit 1
    fi
fi

echo "Starting Ray head node..."
ray start --head \
    --port=$RAY_PORT \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=$DASHBOARD_PORT \
    --num-cpus=2 \
    --num-gpus=0 \
    --include-dashboard=true \
    --disable-usage-stats

sleep 3

# Check if Ray started successfully
if ! ray status > /dev/null 2>&1; then
    echo "❌ Failed to start Ray head node"
    echo "Check logs: /tmp/ray/session_latest/logs/raylet.out"
    exit 1
fi

echo ""
echo "✓ Ray head node started"
echo "  Dashboard: http://$MASTER_HOST:$DASHBOARD_PORT"
echo ""

# Start the coordinator
echo "Starting training coordinator..."

# Check for distributed_master.py in multiple locations
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
sleep 3

# Check if coordinator started
if ! ps -p $COORDINATOR_PID > /dev/null 2>&1; then
    echo "❌ Coordinator failed to start"
    echo "Check the output above for errors"
    ray stop
    exit 1
fi

echo "✓ Coordinator started (PID: $COORDINATOR_PID)"
echo ""

echo "=================================================="
echo "  Master Node Running"
echo "=================================================="
echo ""
echo "✓ Ray Status: ray status"
echo "✓ Dashboard: http://$MASTER_HOST:$DASHBOARD_PORT"
echo "✓ Stop: ray stop"
echo ""
echo "Next steps:"
echo "  1. Start workers: ./start_worker.sh rl_worker_1"
echo "  2. Test: python3 test_distributed_system.py"
echo "  3. Submit jobs: python3 submit_jobs.py --query '...' --episodes 100 --wait"
echo ""
echo "Press Ctrl+C to stop the coordinator"
echo "=================================================="
echo ""

# Wait for coordinator
wait $COORDINATOR_PID
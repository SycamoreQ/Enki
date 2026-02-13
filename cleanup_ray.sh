echo "=================================================="
echo "  Cleaning Up Ray Cluster"
echo "=================================================="
echo ""

# Stop Ray gracefully first
echo "Stopping Ray..."
ray stop 2>/dev/null || true
sleep 2

# Kill any remaining Ray processes
echo "Killing Ray processes..."
pkill -9 -f ray:: 2>/dev/null || true
pkill -9 -f raylet 2>/dev/null || true
pkill -9 -f gcs_server 2>/dev/null || true
pkill -9 -f dashboard 2>/dev/null || true
pkill -9 -f monitor 2>/dev/null || true
pkill -9 -f distributed_master 2>/dev/null || true

sleep 1

# Clean up temp directories
echo "Cleaning Ray temp files..."
rm -rf /tmp/ray/* 2>/dev/null || true

# Check if anything is still running
RAY_PROCS=$(ps aux | grep -E 'ray::|raylet|gcs_server' | grep -v grep | wc -l)

if [ "$RAY_PROCS" -gt 0 ]; then
    echo "Warning: $RAY_PROCS Ray processes still running"
    echo "Run 'ps aux | grep ray' to investigate"
else
    echo "✓ All Ray processes stopped"
fi

echo ""
echo "✓ Cleanup complete"
echo "You can now start Ray with: ./start_master.sh"
echo ""

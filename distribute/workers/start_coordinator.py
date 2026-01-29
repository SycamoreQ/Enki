# distribute/workers/start_coordinator.py

import ray
from distribute.workers.coordinator import TrainingCoordinator

ray.init(address="auto", namespace="distributed_training")

TrainingCoordinator.options(
    name="training_coordinator",
    lifetime="detached",      # 🔥 survives client disconnects
    max_restarts=-1,          # optional
    max_task_retries=-1       # optional
).remote()

print("✓ TrainingCoordinator started and detached")

# Keep process alive if you want logs
import time
while True:
    time.sleep(3600)

import ray

# Connect to the cluster
ray.init(address="ray://172.29.23.125:10001", namespace="distributed_training")

print(f"Current Namespace: {ray.get_runtime_context().namespace}")
print("Searching for actors...")

try:
    # Try to list all actors in this namespace
    actors = ray.util.list_named_actors(all_namespaces=True)
    print(f"All Named Actors: {actors}")
    
    coordinator = ray.get_actor("training_coordinator")
    print("✓ Found it! Coordinator is alive.")
except Exception as e:
    print(f"✗ Failed: {e}")
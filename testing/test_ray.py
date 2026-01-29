import ray
import os
import socket
import time

ray.init(address="auto")

@ray.remote
def worker_task(i):
    return {
        "task": i,
        "hostname": socket.gethostname(),
        "pid": os.getpid()
    }

results = ray.get([worker_task.remote(i) for i in range(8)])

for r in results:
    print(r)

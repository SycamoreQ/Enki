Placeholder README: 

Before starting ray , run these commands to stop and clean cache if the following error occurs:

- /tmp/ray/session_2026-02-03_17-06-36_841240_14060 is over 95% full, available space: 15.1113 GB;         capacity: 440.879 GB. Object creation will fail if spilling is required.

Commands: 

- rm -rf /tmp/ray
- ray stop --force 

After choosing the specific Master and Worker Nodes, on the master we run: 
- ./start_ext_head.sh

This will start Ray on the master and will initialize the cluster with one node inside it i.e the Master. 

to start the Ray orchestrator , which orchestrates jobs: 
python -m distribute.workers.distributed_master --config utils/config/cluster_config.yaml

Then on the separate worker nodes, we run: 

- ray start --address=<HEAD_IP>:<PORT>

This connects to the Ray cluster created by the Master. For the worker to actually become an actor and run jobs: 

- python submit_job.py --curriculum utils/config/curriculum.yaml --stage medium --wait

Will update the README later after training. 
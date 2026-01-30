Placeholder README: 

Before starting ray , run these commands to stop and clean cache: 

rm -rf /tmp/ray
ray stop --force 

After choosing the specific Master and Worker Nodes, on the master we run: 
- ./start_ext_sh 

to start the Ray orchestrator. 

Then on the separate worker nodes, we run: 

- export RAY_ADDRESS=<HEADIP><PORT>
- python -m distribute.workers.submit_job --query "attention mechanisms" --episodes 20 --wait


Will update the README later after training. 
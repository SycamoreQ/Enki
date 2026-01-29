Placeholder README: 

Before starting ray , run these commands to stop and clean cache: 

rm -rf /tmp/ray
ray stop --force 

To start the Ray orchestrator : 

python -m distribute.workers.submit_job --query "attention mechanisms" --episodes 50 --wait

To submit a job: 

export RAY_ADDRESS=<HEADIP><PORT>
python -m distribute.workers.submit_job --query "attention mechanisms" --episodes 20 --wait


Will update the README later after training. 
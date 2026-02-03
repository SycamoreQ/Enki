import ray
import yaml
import time
import argparse
import os

def load_config(config_path='utils/config/cluster_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def submit_staged_jobs(coordinator, curriculum_file):
    """Submit curriculum by difficulty stage"""
    with open(curriculum_file, 'r') as f:
        config = yaml.safe_load(f)
    
    all_jobs = []
    
    for stage_name, jobs in config['stages'].items():
        print(f"STAGE: {stage_name.upper()}")
        stage_jobs = []
        for job_cfg in jobs:
            print(f"{job_cfg['description']}")
            print(f"Query: {job_cfg['query'][:60]}...")
            print(f"Episodes: {job_cfg['episodes']}")
            
            job_id = ray.get(coordinator.submit_job.remote(
                query=job_cfg['query'],
                episodes=job_cfg['episodes'],
                start_paper_id=job_cfg.get('start_paper_id', 'arxiv_1706.03762'),
            ))
            stage_jobs.append({
                'id': job_id,
                'config': job_cfg,
                'submitted': time.time()
            })
            time.sleep(0.5)  # Stagger submissions
        
        all_jobs.extend(stage_jobs)
    
    return all_jobs

def main():
    parser = argparse.ArgumentParser(description='Distributed RL Curriculum Training')
    parser.add_argument('--curriculum', type=str, default='curriculum_jobs.yaml',
                       help='YAML curriculum file')
    parser.add_argument('--stage', type=str, 
                       help='Run specific stage (easy, medium, hard)')
    parser.add_argument('--wait', action='store_true', help='Wait for completion')
    parser.add_argument('--ray-address', type=str, default=os.getenv('RAY_ADDRESS', 'auto'))
    args = parser.parse_args()
    
    ray.init(
        address=args.ray_address ,
        namespace = "distributed_training")
    print(f" Connected to {args.ray_address}")
    
    coordinator = ray.get_actor("training_coordinator",namespace="distributed_training")
    
    # Submit curriculum
    jobs = submit_staged_jobs(coordinator, args.curriculum)
    
    if args.wait:
        print("\nWaiting for jobs to complete...")
        for job_info in jobs:
            # Poll until result available
            while True:
                result = ray.get(coordinator.get_results.remote(job_info['id']))
                if result and result.get('result'):  # Job completed
                    print(f"\n{job_info['config']['description'][:60]}...")
                    print(f"Avg Reward: {result['result']['avg_reward']:.2f}")
                    print(f"Max Sim: {result['result']['max_similarity']:.3f}")
                    break
                print(f"  {job_info['config']['query'][:40]}... waiting...", end='\r')
                time.sleep(5)
    
    ray.shutdown()

if __name__ == "__main__":
    main()

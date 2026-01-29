import ray
import yaml
import time
import argparse


def load_config(config_path='utils/config/cluster_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def submit_single_job(coordinator, query: str, episodes: int, start_paper_id: str):
    """Submit a single training job"""
    job_id = ray.get(coordinator.submit_job.remote(query, episodes, start_paper_id))
    print(f"✓ Job submitted: {job_id}")
    return job_id


def wait_for_job(coordinator, job_id: str, timeout: int = 3600, poll_interval: int = 5):
    """Wait for job completion and return results"""
    print(f"\n⏳ Waiting for job {job_id}...")
    start_time = time.time()
    
    while (time.time() - start_time) < timeout:
        status_info = ray.get(coordinator.get_job_status.remote(job_id))
        status = status_info['status']
        
        print(f"  [{int(time.time() - start_time)}s] Status: {status}", end='\r')
        
        if status == 'completed':
            print(f"\n✓ Job completed in {time.time() - start_time:.1f}s")
            results = ray.get(coordinator.get_results.remote(job_id))
            return results
        
        elif status == 'failed':
            print(f"\n✗ Job failed")
            results = ray.get(coordinator.get_results.remote(job_id))
            return results
        
        time.sleep(poll_interval)
    
    print(f"\nTimeout after {timeout}s")
    return None


def print_results(results):
    """Print job results"""
    if not results:
        return
    
    print("\n" + "="*80)
    print("TRAINING RESULTS")
    print("="*80)
    
    if results['status'] == 'failed':
        print(f"Status: FAILED")
        print(f"Error: {results['error']}")
    else:
        print(f"Status: COMPLETED")
        print(f"Duration: {results['duration']:.1f}s")
        
        if results['result']:
            result = results['result']
            print(f"\nPerformance:")
            print(f"  Episodes: {result.get('episodes', 'N/A')}")
            print(f"  Avg Reward: {result.get('avg_reward', 0):.2f}")
            print(f"  Max Reward: {result.get('max_reward', 0):.2f}")
            print(f"  Avg Similarity: {result.get('avg_similarity', 0):.3f}")
            print(f"  Max Similarity: {result.get('max_similarity', 0):.3f}")
            
            if 'checkpoint_path' in result:
                print(f"\nCheckpoint: {result['checkpoint_path']}")
    
    print("="*80 + "\n")


def submit_batch_jobs(coordinator, jobs_config):
    """Submit multiple jobs from configuration"""
    job_ids = []
    
    for i, job_cfg in enumerate(jobs_config, 1):
        print(f"\nSubmitting job {i}/{len(jobs_config)}:")
        print(f"  Query: {job_cfg['query']}")
        print(f"  Episodes: {job_cfg['episodes']}")
        
        job_id = submit_single_job(
            coordinator,
            job_cfg['query'],
            job_cfg['episodes'],
            job_cfg.get('start_paper_id', 'arxiv_1706.03762')
        )
        job_ids.append(job_id)
        
        time.sleep(1)  # Brief delay between submissions
    
    return job_ids


def main():
    parser = argparse.ArgumentParser(description='Submit distributed training jobs')
    parser.add_argument('--config', type=str, 
                       default='utils/config/cluster_config.yaml',
                       help='Path to cluster configuration')
    parser.add_argument('--query', type=str,
                       help='Research query for training')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes')
    parser.add_argument('--start-paper', type=str, default='arxiv_1706.03762',
                       help='Starting paper ID')
    parser.add_argument('--wait', action='store_true',
                       help='Wait for job completion')
    parser.add_argument('--batch-file', type=str,
                       help='YAML file with batch jobs')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    master_config = config['master_node']
    
    # Connect to Ray cluster
    print("\n" + "="*80)
    print("CONNECTING TO DISTRIBUTED CLUSTER")
    print("="*80)
    print(f"Master: {master_config['host']}:{master_config['ray_port']}")
    print("="*80 + "\n")
    
    ray.init(address="auto", namespace="distributed_training")


    
    print("✓ Connected to cluster")
    print(f"✓ Available resources: {ray.available_resources()}\n")
    
    # Get coordinator actor
    try:
        coordinator = ray.get_actor("training_coordinator", namespace="distributed_training")
    except ValueError:
        print("✗ Coordinator not found. Make sure the master node is running.")
        ray.shutdown()
        return
    
    # Submit jobs
    if args.batch_file:
        # Batch submission
        with open(args.batch_file, 'r') as f:
            batch_config = yaml.safe_load(f)
        
        job_ids = submit_batch_jobs(coordinator, batch_config['jobs'])
        
        if args.wait:
            print(f"\nWaiting for {len(job_ids)} jobs to complete...")
            for job_id in job_ids:
                results = wait_for_job(coordinator, job_id)
                print_results(results)
    
    else:
        # Single job submission
        if not args.query:
            print("✗ --query required for single job submission")
            ray.shutdown()
            return
        
        job_id = submit_single_job(
            coordinator,
            args.query,
            args.episodes,
            args.start_paper
        )
        
        if args.wait:
            results = wait_for_job(coordinator, job_id)
            print_results(results)
    
    # Print current system status
    status = ray.get(coordinator.get_status.remote())
    print("\nCurrent System Status:")
    print(f"  Pending: {status['jobs']['pending']}")
    print(f"  Running: {status['jobs']['running']}")
    print(f"  Completed: {status['jobs']['completed']}")
    
    ray.shutdown()
    print("\n✓ Done\n")


if __name__ == "__main__":
    main()
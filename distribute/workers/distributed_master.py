import ray
import yaml
import asyncio
import time
from typing import List, Dict
from dataclasses import dataclass, field
from collections import deque


@dataclass
class TrainingJob:
    """Training job specification"""
    job_id: str
    query: str
    episodes: int
    start_paper_id: str
    worker_id: str = None
    status: str = "pending"  # pending, running, completed, failed
    submitted_at: float = field(default_factory=time.time)
    started_at: float = None
    completed_at: float = None
    result: Dict = None
    error: str = None


class RoundRobinScheduler:
    
    def __init__(self, worker_ids: List[str]):
        self.workers = deque(worker_ids)
        self.job_counts = {wid: 0 for wid in worker_ids}
    
    def assign_next_worker(self) -> str:
        """Assign job to next worker in round-robin fashion"""
        if not self.workers:
            return None
        
        # Rotate to next worker
        self.workers.rotate(-1)
        worker = self.workers[0]
        self.job_counts[worker] += 1
        return worker
    
    def get_stats(self) -> Dict:
        """Get scheduler statistics"""
        return {
            'total_workers': len(self.workers),
            'job_distribution': self.job_counts.copy()
        }


@ray.remote
class DistributedTrainingCoordinator:
    """
    Master coordinator for distributed RL training.
    Manages job queue and worker assignments.
    """
    
    def __init__(self, config_path: str = "utils/config/cluster_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize workers from config
        self.workers = {}
        for worker_config in self.config['rl_training_workers']:
            if worker_config['host']: 
                self.workers[worker_config['name']] = {
                    'status': 'idle',
                    'current_job': None,
                    'jobs_completed': 0,
                    'jobs_failed': 0,
                    'config': worker_config
                }
        
        worker_ids = list(self.workers.keys())
        self.scheduler = RoundRobinScheduler(worker_ids)
        
        # Job tracking
        self.pending_jobs = []
        self.running_jobs = {}
        self.completed_jobs = {}
        self.job_futures = {}
        
        print(f"✓ Coordinator initialized with {len(self.workers)} workers")
        for wid in worker_ids:
            print(f"  - {wid}")
    
    def submit_job(self, query: str, episodes: int, start_paper_id: str) -> str:
        """Submit a training job"""
        job_id = f"job_{int(time.time() * 1000)}"
        
        job = TrainingJob(
            job_id=job_id,
            query=query,
            episodes=episodes,
            start_paper_id=start_paper_id
        )
        
        self.pending_jobs.append(job)
        print(f"Job submitted: {job_id}")
        print(f"Query: {query[:50]}...")
        print(f"Episodes: {episodes}")
        
        return job_id
    
    def dispatch_job(self, job: TrainingJob) -> bool:
        worker_id = self.scheduler.assign_next_worker()
        
        if not worker_id or self.workers[worker_id]['status'] != 'idle':
            return False
        
        # Get database config
        db_config = self.config['database_server']
        
        worker_config = self.workers[worker_id]['config']
        
        try:
            # Try to import real trainer, fall back to mock
            try:
                from RL.train_rl import DistributedRLTrainer
                print(f"Using real DistributedRLTrainer for {worker_id}")
            except (ImportError, ModuleNotFoundError) as e:
                print(f"Real trainer not available ({e}), using mock trainer")
                from workers.mock_trainer import MockRLTrainer as DistributedRLTrainer
            
            # Create remote trainer actor
            trainer = DistributedRLTrainer.options(
                name=f"trainer_{worker_id}_{job.job_id}",
                num_cpus=min(worker_config['resources']['CPU'], 1), 
                num_gpus=0,  # No GPU in local mode
                memory=min(worker_config['resources']['memory'], 1000000000)  # 1GB max
            ).remote(
                worker_id=worker_id,
                db_host=db_config.get('host', 'localhost'),
                db_port=db_config.get('neo4j_port', 7687),
                db_user=db_config.get('neo4j_user', 'neo4j'),
                db_password=db_config.get('neo4j_password', ''),
                db_name=db_config.get('database_name', 'neo4j')
            )
            
            # Start training
            training_config = self.config.get('training_config', {})
            
            future = trainer.train_episodes.remote(
                episodes=job.episodes,
                query=job.query,
                start_paper_id=job.start_paper_id,
                **training_config
            )
            
            # Update job and worker state
            job.worker_id = worker_id
            job.status = "running"
            job.started_at = time.time()
            
            self.workers[worker_id]['status'] = 'busy'
            self.workers[worker_id]['current_job'] = job.job_id
            
            self.running_jobs[job.job_id] = job
            self.job_futures[job.job_id] = future
            
            print(f"Job dispatched: {job.job_id} → {worker_id}")
            return True
            
        except Exception as e:
            print(f"Failed to dispatch job {job.job_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def schedule_jobs(self):
        """Schedule pending jobs to available workers"""
        scheduled_count = 0
        
        while self.pending_jobs:
            job = self.pending_jobs[0]
            
            if self.dispatch_job(job):
                self.pending_jobs.pop(0)
                scheduled_count += 1
            else:
                break  # No available workers
        
        return scheduled_count
    
    def check_completed_jobs(self) -> List[str]:
        """Check for completed jobs and update state"""
        completed_job_ids = []
        
        for job_id, future in list(self.job_futures.items()):
            # Check if job is complete
            ready, _ = ray.wait([future], timeout=0)
            
            if ready:
                job = self.running_jobs.pop(job_id)
                del self.job_futures[job_id]
                
                try:
                    result = ray.get(ready[0])
                    job.status = "completed"
                    job.completed_at = time.time()
                    job.result = result
                    
                    # Update worker
                    worker_id = job.worker_id
                    self.workers[worker_id]['status'] = 'idle'
                    self.workers[worker_id]['current_job'] = None
                    self.workers[worker_id]['jobs_completed'] += 1
                    
                    duration = job.completed_at - job.started_at
                    print(f"Job completed: {job_id} ({duration:.1f}s)")
                    print(f"Worker: {worker_id}")
                    print(f"Avg reward: {result.get('avg_reward', 0):.2f}")
                    print(f"Max similarity: {result.get('max_similarity', 0):.3f}")
                    
                except Exception as e:
                    job.status = "failed"
                    job.completed_at = time.time()
                    job.error = str(e)
                    
                    # Update worker
                    worker_id = job.worker_id
                    self.workers[worker_id]['status'] = 'idle'
                    self.workers[worker_id]['current_job'] = None
                    self.workers[worker_id]['jobs_failed'] += 1
                    
                    print(f"✗ Job failed: {job_id}")
                    print(f"  Error: {e}")
                
                self.completed_jobs[job_id] = job
                completed_job_ids.append(job_id)
        
        return completed_job_ids
    
    def get_status(self) -> Dict:
        """Get system status"""
        return {
            'workers': {
                wid: {
                    'status': info['status'],
                    'current_job': info['current_job'],
                    'jobs_completed': info['jobs_completed'],
                    'jobs_failed': info['jobs_failed']
                }
                for wid, info in self.workers.items()
            },
            'jobs': {
                'pending': len(self.pending_jobs),
                'running': len(self.running_jobs),
                'completed': len(self.completed_jobs)
            },
            'scheduler': self.scheduler.get_stats()
        }
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get status of a specific job"""
        if job_id in self.running_jobs:
            return {'status': 'running', 'job': self.running_jobs[job_id]}
        elif job_id in self.completed_jobs:
            return {'status': 'completed', 'job': self.completed_jobs[job_id]}
        else:
            for job in self.pending_jobs:
                if job.job_id == job_id:
                    return {'status': 'pending', 'job': job}
        return {'status': 'not_found', 'job': None}
    
    def get_results(self, job_id: str) -> Dict:
        """Get results of a completed job"""
        if job_id in self.completed_jobs:
            job = self.completed_jobs[job_id]
            return {
                'status': job.status,
                'result': job.result,
                'duration': job.completed_at - job.started_at if job.completed_at else None,
                'error': job.error
            }
        return None


async def run_master_loop(coordinator, duration_sec: float = None):
    print("STARTING DISTRIBUTED TRAINING COORDINATOR")
    print(f"Duration: {'infinite' if duration_sec is None else f'{duration_sec}s'}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    try:
        while True:
            # Schedule pending jobs
            scheduled = ray.get(coordinator.schedule_jobs.remote())
            if scheduled > 0:
                print(f"Scheduled {scheduled} jobs")
            
            # Check completed jobs
            completed = ray.get(coordinator.check_completed_jobs.remote())
            
            # Print status every 10 seconds
            if int(time.time()) % 10 == 0:
                status = ray.get(coordinator.get_status.remote())
                print(f"\n[{time.strftime('%H:%M:%S')}] System Status:")
                print(f"  Jobs: {status['jobs']['running']} running, "
                      f"{status['jobs']['pending']} pending, "
                      f"{status['jobs']['completed']} completed")
                
                for wid, info in status['workers'].items():
                    print(f"  {wid}: {info['status']} "
                          f"(completed: {info['jobs_completed']}, "
                          f"failed: {info['jobs_failed']})")
            
            # Check duration
            if duration_sec and (time.time() - start_time) >= duration_sec:
                print("\nCoordination loop duration complete")
                break
            
            await asyncio.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nCoordination interrupted")


async def main():
    import argparse
    import sys
    import os
    
    # Add parent directory to path if needed
    if os.path.exists('distribute/workers'):
        sys.path.insert(0, os.path.abspath('.'))
    
    parser = argparse.ArgumentParser(description='Start distributed training master')
    parser.add_argument('--config', type=str, 
                       default='utils/config/cluster_config.yaml',
                       help='Path to cluster configuration')
    parser.add_argument('--duration', type=int, default=None,
                       help='Run duration in seconds')
    
    args = parser.parse_args()
    
    # Read config to get master settings
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    master_config = config['master_node']
    
    # Initialize Ray head node
    print("INITIALIZING RAY HEAD NODE")
    print(f"Host: {master_config['host']}")
    print(f"Port: {master_config['ray_port']}")
    print(f"Dashboard: {master_config['dashboard_port']}")
    print("="*80 + "\n")
    
    ray.init(
        dashboard_host='0.0.0.0',
        dashboard_port=master_config['dashboard_port'],
        include_dashboard=True,
        namespace='distributed_training'  # Use named namespace
    )
    
    print("Ray head node started")
    print(f"Dashboard: http://{master_config['host']}:{master_config['dashboard_port']}")
    print(f"Cluster resources: {ray.cluster_resources()}\n")
    
    # Create coordinator actor
    coordinator = DistributedTrainingCoordinator.options(
        name="training_coordinator",
        namespace="distributed_training",
        lifetime="detached",
        max_restarts=-1
    ).remote(config_path=args.config)
    
    # Wait for workers to connect
    print("Waiting for workers to connect")
    await asyncio.sleep(15)
    
    # Run coordination loop
    await run_master_loop(coordinator, args.duration)
    
    # Print final status
    status = ray.get(coordinator.get_status.remote())
    print("FINAL STATUS")
    print(f"Total jobs completed: {status['jobs']['completed']}")
    print(f"Worker job distribution:")
    for wid, count in status['scheduler']['job_distribution'].items():
        print(f"  {wid}: {count} jobs")
    print("="*80 + "\n")
    
    print("Master node will continue running. Press Ctrl+C to stop.")
    print("Workers can continue submitting jobs via the coordinator actor.\n")
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        print("\nShutting down...")
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
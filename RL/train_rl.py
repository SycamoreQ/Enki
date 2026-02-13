import ray
import time
import asyncio
import numpy as np
import pickle
import os
from collections import defaultdict
from typing import Dict
from RL.ddqn import DDQLAgent
from RL.env import AdvancedGraphTraversalEnv, RelationType
from graph.database.store import EnhancedStore
from RL.curriculum import CurriculumManager

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBLogger:
    def __init__(self, enabled=True):
        self.enabled = enabled and WANDB_AVAILABLE
        self.run = None

    def init(self, **kwargs):
        if not self.enabled:
            return self
        try:
            self.run = wandb.init(**kwargs)
            return self.run
        except Exception as e:
            print(f"W&B init failed: {e}")
            self.enabled = False
            return self

    def log(self, metrics):
        if self.enabled and self.run:
            try:
                wandb.log(metrics)
            except:
                pass

    def finish(self):
        if self.enabled and self.run:
            try:
                wandb.finish()
            except:
                pass


DEFAULT_CONFIG = {
    "total_episodes": 10000,
    "max_steps_per_episode": 12,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "gamma": 0.95,
    "epsilon_start": 1.0,
    "epsilon_min": 0.15,
    "epsilon_decay": 0.99985,
    "epsilon_warmup_episodes": 100,
    "target_update_freq": 10,
    "use_communities": True,
    "state_dim": 783,
    "text_dim": 384,
    "use_prioritized_replay": True,
    "train_every_n_steps": 2,
    "end_of_episode_replays": 3,
}


def normalize_paper_id(paper_id: str) -> str:
    if not paper_id:
        return ""
    paper_id = str(paper_id).strip().lstrip('0')
    return paper_id if paper_id else "0"


def get_available_cache_relations(env, pid: str):
    """Return list of RelationType enums that have edges in training cache."""
    pid = normalize_paper_id(pid)
    edges = env.training_edge_cache.get(pid, [])
    etypes = {et for et, _ in edges}
    actions = []
    if "cites" in etypes:
        actions.append(RelationType.CITES)
    if "citedby" in etypes:
        actions.append(RelationType.CITED_BY)
    return actions


async def load_training_cache():
    """Load cached training data."""
    print("Loading training cache...")
    cache_dir = 'training_cache'
    
    with open(os.path.join(cache_dir, 'training_papers_1M.pkl'), 'rb') as f:
        papers = pickle.load(f)
    print(f"Loaded {len(papers):,} papers")

    with open(os.path.join(cache_dir, 'edge_cache_1M.pkl'), 'rb') as f:
        edge_cache = pickle.load(f)
    print(f"Loaded edge cache")

    with open(os.path.join(cache_dir, 'paper_id_set_1M.pkl'), 'rb') as f:
        paper_id_set = pickle.load(f)
    print(f"Loaded paper ID index")

    return papers, edge_cache, paper_id_set


async def build_embeddings():
    """Build or load embeddings."""
    papers, edge_cache, paper_id_set = await load_training_cache()
    
    print("Loading embeddings...")
    from utils.batchencoder import BatchEncoder
    encoder = BatchEncoder(
        model_name='all-MiniLM-L6-v2',
        batch_size=256,
        cache_file='training_cache/embeddings_1M.pkl'
    )
    
    encoder.precompute_paper_embeddings(papers, force=False)
    embeddings_raw = encoder.cache

    # Normalize IDs
    embeddings = {normalize_paper_id(str(k)): v for k, v in embeddings_raw.items()}

    # Normalize edge cache
    edge_cache_str = {}
    for src, edges in edge_cache.items():
        src_normalized = normalize_paper_id(str(src))
        normalized_edges = [
            (et, normalize_paper_id(str(tid)))
            for et, tid in edges
        ]
        edge_cache_str[src_normalized] = normalized_edges

    print(f"Loaded {len(embeddings):,} embeddings")
    return papers, edge_cache_str, paper_id_set, embeddings, encoder


@ray.remote(num_gpus=1)
class DistributedRLTrainer:
    """
    Distributed RL trainer for parallel training with curriculum learning.
    Each worker trains independently on remote database.
    Called by the coordinator to execute training jobs.
    """
    
    def __init__(
        self,
        worker_id: str,
        db_host: str,
        db_port: int = 7687,
        db_user: str = "neo4j",
        db_password: str = "password",
        db_name: str = "neo4j"
    ):
        self.worker_id = worker_id
        self.db_config = {
            'host': db_host,
            'port': db_port,
            'user': db_user,
            'password': db_password,
            'database': db_name
        }
        self._initialized = False
        
        print(f"[{self.worker_id}] DistributedRLTrainer initialized")
        print(f"  Database: {db_host}:{db_port}/{db_name}")
    
    async def _initialize_training_environment(self):
        if self._initialized:
            return

        print(f"[{self.worker_id}] Initializing training environment...")

        # Load data
        self.papers, self.edge_cache, self.paper_id_set, self.embeddings, self.encoder = await build_embeddings()

        # Connect to database
        self.store = EnhancedStore(pool_size=5)

        # Initialize environment
        self.env = AdvancedGraphTraversalEnv(
            self.store,
            precomputed_embeddings=self.embeddings,
            use_communities=DEFAULT_CONFIG['use_communities'],
            use_feedback=False,
            use_query_parser=True,
            parser_type='dspy',
            require_precomputed_embeddings=False
        )

        # Setup training cache
        embedded_ids = set(self.embeddings.keys())
        normalized_paper_id_set = {normalize_paper_id(str(pid)) for pid in self.paper_id_set}
        self.env.training_paper_ids = normalized_paper_id_set

        # Prune edge cache
        pruned_edge_cache = {}
        for src, edges in self.edge_cache.items():
            src = normalize_paper_id(str(src))
            if src not in embedded_ids:
                continue
            
            kept = [
                (et.lower().replace("_", ""), normalize_paper_id(str(tid)))
                for et, tid in edges
                if et.lower().replace("_", "") in ("cites", "citedby") and
                   normalize_paper_id(str(tid)) in embedded_ids
            ]
            
            if kept:
                pruned_edge_cache[src] = kept

        self.env.training_edge_cache = pruned_edge_cache
        self.env.precomputed_embeddings = self.embeddings
        self.embedded_ids = embedded_ids

        print(f"[{self.worker_id}] ✓ Environment initialized")
        print(f"  Papers: {len(self.env.training_paper_ids):,}")
        print(f"  Edges: {sum(len(v) for v in self.env.training_edge_cache.values()):,}")

        self._initialized = True

    async def train_episodes(
        self,
        episodes: int,
        query: str,
        start_paper_id: str,
        **kwargs
    ) -> Dict:
        """
        Train RL agent for specified episodes with curriculum learning.
        
        Args:
            episodes: Number of training episodes
            query: Base query (used for Stage 1, curriculum generates others)
            start_paper_id: Starting paper for episode 0
            **kwargs: Additional training config
        
        Returns:
            Dict with training results + curriculum statistics
        """
        start_time = time.time()
        
        # Merge config
        CONFIG = {**DEFAULT_CONFIG, **kwargs}
        CONFIG['total_episodes'] = episodes
        
        print(f"\n[{self.worker_id}] Starting Distributed RL Training")
        print(f"  Episodes: {episodes}")
        print(f"  Base Query: {query[:50]}...")
        print(f"  Start paper: {start_paper_id}")
        
        # Initialize environment
        await self._initialize_training_environment()
        
        # Initialize WandB logger
        logger = WandBLogger(enabled=kwargs.get('use_wandb', False))
        if logger.enabled:
            logger.init(
                project="Enki",
                config=CONFIG,
                name=f"{self.worker_id}_{time.strftime('%Y%m%d_%H%M%S')}",
                tags=["distributed", "ddqn", "curriculum", self.worker_id]
            )

        # Initialize curriculum manager with base query context
        curriculum = CurriculumManager(self.papers, self.encoder)

        import torch

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(f"[{self.worker_id}] DDQN Agent initialized on GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print(f"[{self.worker_id}] DDQN Agent initialized on CPU")
        
        # Initialize agent
        agent = DDQLAgent(
            state_dim=CONFIG['state_dim'],
            text_dim=CONFIG['text_dim'],
            use_prioritized=CONFIG['use_prioritized_replay'],
            precomputed_embeddings=self.embeddings
        )
        
        print(f"[{self.worker_id}] DDQN Agent initialized on cpu")
        print(f"Prioritized Replay: {CONFIG['use_prioritized_replay']}")
        print(f"Replay Buffer Size: 200,000 | Warmup: 32 | Batch: 32")

        # Training state tracking
        episode_rewards = []
        episode_similarities = []
        episode_lengths = []
        dead_end_count = 0
        success_count = 0
        total_training_steps = 0
        stage_stats = defaultdict(list) 

        # Main curriculum-aware training loop
        for episode in range(episodes):
            try:
                # Get current curriculum stage
                stage = curriculum.get_current_stage(episode)
                
                if episode == 0:
                    episode_query = query
                    episode_start_paper_id = normalize_paper_id(start_paper_id)
                else:
                    # Curriculum selects query + optimal starting paper for this stage
                    episode_query = curriculum.get_query_for_stage(stage, episode)
                    start_paper = curriculum.get_starting_paper(episode_query, stage)
                    episode_start_paper_id = normalize_paper_id(
                        str(start_paper.get('paperId') or start_paper.get('paper_id'))
                    )
                
                # Log curriculum progression
                stage_name = stage['name'].split(':')[1].strip()[:20]
                print(f"[{self.worker_id}] Ep {episode:3d}/{episodes} | "
                      f"Stage: {stage_name:<15} | Steps: {stage['max_steps']:2d} | "
                      f"Query: {episode_query[:40]}...")
                
                # Validate starting paper exists in cache
                if episode_start_paper_id not in self.env.training_edge_cache:
                    print(f"  → Skipping: start paper {episode_start_paper_id} not in cache")
                    episode_rewards.append(0.0)
                    episode_similarities.append(0.0)
                    episode_lengths.append(0)
                    continue

                # Validate neighbors exist
                neighbor_ids = [tid for _, tid in self.env.training_edge_cache[episode_start_paper_id]]
                if not any(nid in self.embedded_ids for nid in neighbor_ids):
                    print(f"Skipping: no valid neighbors for {episode_start_paper_id}")
                    episode_rewards.append(0.0)
                    episode_similarities.append(0.0)
                    episode_lengths.append(0)
                    continue

                # Reset environment with curriculum parameters
                state = await self.env.reset(
                    episode_query, 
                    intent=1, 
                    start_node_id=episode_start_paper_id
                )
                max_steps = min(stage['max_steps'], CONFIG['max_steps_per_episode'])

                # Run episode
                episode_reward = 0
                steps = 0
                step_losses = []

                for step in range(max_steps):
                    # Manager step (hierarchical RL)
                    pid = normalize_paper_id(
                        str(self.env.current_node.get("paperId") or self.env.current_node.get("paper_id"))
                    )
                    available = get_available_cache_relations(self.env, pid)
                    if not available:
                        break

                    relation_type = int(np.random.choice(available))
                    is_terminal, manager_reward = await self.env.manager_step(relation_type)
                    episode_reward += manager_reward

                    if is_terminal:
                        break

                    # Worker step (DDQN agent)
                    worker_actions = await self.env.get_worker_actions()
                    if not worker_actions:
                        break

                    # Filter valid actions (cache + embeddings)
                    worker_actions = [
                        (n, r) for (n, r) in worker_actions
                        if normalize_paper_id(
                            str(n.get("paperid") or n.get("paperId") or n.get("paper_id"))
                        ) in self.embedded_ids
                    ][:15]

                    if not worker_actions:
                        break

                    # Agent selects best action
                    best_action = agent.act(state, worker_actions, max_actions=15)
                    if not best_action or not isinstance(best_action, tuple):
                        break

                    chosen_node, chosen_relation = best_action

                    # Execute action
                    next_state, worker_reward, done = await self.env.worker_step(chosen_node)
                    
                    # Curriculum-aware exploration bonus
                    exploration_bonus = 0.0
                    if steps >= 5:
                        exploration_bonus = 0.5 * (steps / max_steps) * stage['start_similarity_threshold']
                    
                    total_reward = worker_reward + exploration_bonus
                    episode_reward += total_reward
                    steps += 1

                    # Store transition for prioritized replay
                    next_actions = await self.env.get_worker_actions() if not done else []
                    next_actions = [
                        (n, r) for n, r in next_actions
                        if normalize_paper_id(str(n.get('paperId') or n.get('paper_id'))) in self.embedded_ids
                    ][:15]

                    agent.remember(
                        state=state,
                        action_tuple=best_action,
                        reward=total_reward,
                        next_state=next_state,
                        done=done,
                        next_actions=next_actions
                    )
                    
                    # Train agent
                    if len(agent.memory) >= agent.batch_size and step % CONFIG['train_every_n_steps'] == 0:
                        loss = agent.replay()
                        step_losses.append(loss)
                        total_training_steps += 1

                        # Epsilon decay (post-warmup)
                        if episode >= CONFIG['epsilon_warmup_episodes']:
                            agent.epsilon = max(
                                CONFIG['epsilon_min'],
                                agent.epsilon * CONFIG['epsilon_decay']
                            )

                    state = next_state
                    if done:
                        break
                
                # End-of-episode prioritized replay
                if len(agent.memory) >= agent.batch_size:
                    for _ in range(CONFIG['end_of_episode_replays']):
                        loss = agent.replay()
                        step_losses.append(loss)
                        total_training_steps += 1

                        if episode >= CONFIG['epsilon_warmup_episodes']:
                            agent.epsilon = max(
                                CONFIG['epsilon_min'],
                                agent.epsilon * CONFIG['epsilon_decay']
                            )

                # Update target network
                if episode % CONFIG['target_update_freq'] == 0 and episode > 0:
                    agent.update_target()

                # Track episode metrics
                episode_loss = np.mean(step_losses) if step_losses else 0.0
                final_sim = getattr(self.env, 'best_similarity_so_far', 0.0)

                # Stage-specific performance tracking
                stage_key = stage['query_difficulty']
                stage_stats[stage_key].append({
                    'reward': episode_reward,
                    'similarity': final_sim,
                    'steps': steps
                })

                # Global episode tracking
                if steps < 2:
                    dead_end_count += 1

                if final_sim > 0.5:
                    success_count += 1

                episode_rewards.append(episode_reward)
                episode_similarities.append(final_sim if final_sim > -0.5 else np.nan)
                episode_lengths.append(steps)

                # Update curriculum performance
                curriculum.update_performance(episode_reward, final_sim)

                # Log to Weights & Biases
                if logger.enabled:
                    logger.log({
                        "episode": episode,
                        "stage": stage['query_difficulty'],
                        "episode_reward": episode_reward,
                        "episode_similarity": final_sim,
                        "episode_steps": steps,
                        "epsilon": agent.epsilon,
                        "loss": episode_loss,
                        "max_steps": stage['max_steps']
                    })

                # Progress reporting every 10 episodes
                if episode % 10 == 0 or episode < 5:
                    recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
                    recent_sims = episode_similarities[-10:] if len(episode_similarities) >= 10 else episode_similarities
                    
                    avg_reward = float(np.mean(recent_rewards) if recent_rewards else 0.0)
                    avg_sim = float(np.nanmean(recent_sims) if recent_sims else 0.0)
                    
                    print(f"[{self.worker_id}] Ep {episode:4d}/{episodes} | "
                          f"R: {episode_reward:+6.2f} | "
                          f"A10R: {avg_reward:+6.2f} | "
                          f"S: {final_sim:.3f} | "
                          f"ε: {agent.epsilon:.3f} | "
                          f"L: {steps}/{stage['max_steps']}")
            
            except Exception as e:
                print(f"[{self.worker_id}] Episode {episode} failed: {str(e)[:100]}")
                # Record failed episode
                episode_rewards.append(0.0)
                episode_similarities.append(0.0)
                episode_lengths.append(0)
                continue
        
        duration = time.time() - start_time
        
        # Save curriculum-aware checkpoint
        checkpoint_dir = f"checkpoints/{self.worker_id}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = f"{checkpoint_dir}/curriculum_final_ep{episodes}.pt"
        agent.save(checkpoint_path)
        
        # Compute comprehensive results
        safe_rewards = episode_rewards if episode_rewards else [0.0]
        safe_sims = episode_similarities if episode_similarities else [0.0]
        
        result = {
            'job_type': 'distributed_rl_curriculum_training',
            'worker_id': self.worker_id,
            'episodes': episodes,
            'duration_sec': duration,
            'avg_reward': float(np.mean(safe_rewards)),
            'max_reward': float(np.max(safe_rewards)),
            'final_reward': float(safe_rewards[-1]),
            'avg_similarity': float(np.nanmean(safe_sims)),
            'max_similarity': float(np.nanmax(safe_sims)),
            'final_similarity': float(safe_sims[-1]),
            'success_rate': 100 * success_count / max(episodes, 1),
            'dead_end_rate': 100 * dead_end_count / max(episodes, 1),
            'avg_episode_length': float(np.mean(episode_lengths)),
            'curriculum_stats': dict(stage_stats),
            'checkpoint_path': checkpoint_path,
            'status': 'completed'
        }
        
        # Final summary
        print(f"\n[{self.worker_id}] Training Complete")
        print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"  Total Steps: {total_training_steps:,}")
        print(f"  Avg Reward: {result['avg_reward']:.2f}")
        print(f"  Max Similarity: {result['max_similarity']:.3f}")
        print(f"  Success Rate: {result['success_rate']:.1f}%")
        print(f"  Checkpoint: {checkpoint_path}")
        
        # Stage-wise performance
        print(f"\nStage Performance:")
        for stage_name, stats in result['curriculum_stats'].items():
            if stats:
                avg_reward = np.mean([s['reward'] for s in stats])
                avg_sim = np.mean([s['similarity'] for s in stats])
                print(f"  {stage_name.upper()}: R={avg_reward:.2f}, Sim={avg_sim:.3f} (n={len(stats)})")
        
        logger.finish()
        return result
    
    def __del__(self):
        """Cleanup database connections."""
        if hasattr(self, 'store') and self.store:
            try:
                import asyncio
                asyncio.run(self.store.pool.close())
            except:
                pass

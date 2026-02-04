import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from .prioritized_replay import PrioritizedReplay
from .noisy_layer import NoisyLinear
from .n_step import NStepBuffer



class QNetwork(nn.Module):
    def __init__(self, state_dim, action_emb_dim, relation_dim=13):
        super(QNetwork, self).__init__()
        
        self.input_dim = state_dim + action_emb_dim + relation_dim
        
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, 1)
        )
    
    def forward(self, state, action_emb, relation_onehot):
        x = torch.cat([state, action_emb, relation_onehot], dim=1)
        return self.network(x)
    
    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class DDQLAgent:
    def __init__(self, state_dim=783, text_dim=384, use_prioritized=True, precomputed_embeddings=None):
        self.state_dim = state_dim
        self.text_dim = text_dim
        self.relation_dim = 13
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = QNetwork(state_dim, text_dim, self.relation_dim).to(self.device)
        self.target_net = QNetwork(state_dim, text_dim, self.relation_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=500, T_mult=2, eta_min=1e-6
        )
        
        self.use_prioritized = use_prioritized
        if use_prioritized:
            self.memory = PrioritizedReplay(capacity=200000, alpha=0.6, beta=0.4)
        else:
            self.memory = deque(maxlen=200000)
        
        self.batch_size = 32
        self.gamma = 0.99 

        self.epsilon = 0.95        
        self.epsilon_min = 0.10      
        self.epsilon_decay = 0.0017   
        
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.precomputed_embeddings = precomputed_embeddings or {}
        
        self.n_step_buffer = NStepBuffer(n=1, gamma=self.gamma)  
        
        self.training_step = 0
        self.warmup_steps = 32  
        
        print(f"DDQN Agent initialized on {self.device}")
        print(f"Prioritized Replay: {use_prioritized}")
        print(f"Replay Buffer Size: 200,000")
        print(f"Warmup Steps: {self.warmup_steps:,}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Precomputed embeddings: {len(self.precomputed_embeddings):,}")
    
    def _get_relation_onehot(self, r_type: int) -> torch.Tensor:
        onehot = torch.zeros(1, self.relation_dim)
        if 0 <= r_type < self.relation_dim:
            onehot[0][r_type] = 1.0
        else:
            onehot[0][0] = 1.0
        return onehot
    
    def _encode_node(self, node: Dict) -> np.ndarray:
        paper_id = node.get('paper_id') or node.get('paperId')

        if paper_id and paper_id in self.precomputed_embeddings:
            return self.precomputed_embeddings[paper_id]

        title = node.get('title') or ''
        abstract = node.get('abstract') or ''

        title = str(title)
        abstract = str(abstract)

        text = f"{title} {abstract[:200]}" if abstract else title

        if not text.strip():
            if paper_id:
                text = f"Paper {paper_id}"
            else:
                return np.zeros(self.text_dim)

        return self.encoder.encode(text, convert_to_numpy=True)
    
    def act(self, state, actions, max_actions=5):
        if not actions:
            print("    [AGENT] No actions provided")
            return None
        
        # Validate all actions
        valid_actions = []
        for i, action in enumerate(actions):
            if not isinstance(action, tuple) or len(action) != 2:
                print(f"    [AGENT] Skipping invalid action at index {i}: {type(action)}")
                continue
            node, relation_type = action
            if not isinstance(node, dict):
                print(f"    [AGENT] Skipping action with invalid node at index {i}: {type(node)}")
                continue
            valid_actions.append(action)
        
        if not valid_actions:
            print("    [AGENT] No valid actions after filtering")
            return None
        
        actions = valid_actions[:max_actions]
        
        print(f"    [AGENT] act() called with {len(actions)} actions, max={max_actions}")
        
        if np.random.random() < self.epsilon:
            chosen_idx = random.randint(0, len(actions) - 1)
            chosen_action = actions[chosen_idx]
            print(f"    [AGENT] Random action (ε={self.epsilon:.3f})")
            
            if not isinstance(chosen_action, tuple) or len(chosen_action) != 2:
                print(f"    [AGENT] ERROR: Random selection returned malformed action: {type(chosen_action)}")
                return actions[0]
            
            return chosen_action
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        q_values = []
        valid_action_indices = []
        
        for idx, action_tuple in enumerate(actions):
            try:
                node, relation_type = action_tuple
                
                # Get node embedding
                node_id = str(node.get('paperId') or node.get('paper_id') or '')
                
                if node_id and node_id in self.precomputed_embeddings:
                    node_emb = self.precomputed_embeddings[node_id]
                else:
                    node_emb = self._encode_node(node)
                
                if not isinstance(node_emb, np.ndarray):
                    node_emb = np.array(node_emb, dtype=np.float32)
                
                if node_emb.shape[0] != self.text_dim:
                    print(f"    [AGENT] Warning: node embedding has wrong shape {node_emb.shape}, expected ({self.text_dim},)")
                    node_emb = np.zeros(self.text_dim, dtype=np.float32)
                
                node_emb_tensor = torch.FloatTensor(node_emb).unsqueeze(0).to(self.device)
                relation_onehot = self._get_relation_onehot(relation_type).to(self.device)
                
                with torch.no_grad():
                    q_val = self.policy_net(state_tensor, node_emb_tensor, relation_onehot).item()
                
                q_values.append(q_val)
                valid_action_indices.append(idx)
                
            except Exception as e:
                print(f"    [AGENT] Error processing action {idx}: {e}")
                continue
        
        if not q_values:
            print("    [AGENT] No valid Q-values computed, selecting random action")
            chosen_idx = random.randint(0, len(actions) - 1)
            return actions[chosen_idx]
        
        print(f"    [AGENT] Processed {len(q_values)}/{len(actions)} actions")
        
        best_idx = valid_action_indices[np.argmax(q_values)]
        best_q = max(q_values)
        
        print(f"    [AGENT] Best Q={best_q:.3f}")
        
        chosen_action = actions[best_idx]
        
        if not isinstance(chosen_action, tuple) or len(chosen_action) != 2:
            print(f"    [AGENT] ERROR: chosen_action is malformed: {type(chosen_action)}")
            return actions[0] 
        
        return chosen_action



    def remember(self, state, action_tuple, reward, next_state, done, next_actions):
        node, r_type = action_tuple
        act_emb = self._encode_node(node)
        
        n_step_transition = self.n_step_buffer.add(
            state, act_emb, r_type, reward, next_state, done, next_actions
        )
        
        if n_step_transition:
            print(f"  [MEM] Adding transition to memory (size: {len(self.memory)})")
            if self.use_prioritized:
                self.memory.add(*n_step_transition)
            else:
                self.memory.append(n_step_transition)
        
        if done:
            for trans in self.n_step_buffer.flush():
                print(f"  [MEM] Episode done, flushing transitions")
                if self.use_prioritized:
                    self.memory.add(*trans)
                else:
                    self.memory.append(trans)

    def replay_prioritized(self) -> float:
        if len(self.memory) < self.warmup_steps:
            return 0.0
            
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch, weights, indices = self.memory.sample(self.batch_size)
        indices = [int(idx) for idx in indices]
        
        if not batch:
            return 0.0
        
        states = []
        action_embs = []
        relation_onehots = []
        rewards = []
        next_states = []
        dones = []
        next_action_embs = []
        next_relation_onehots = []
        
        self.policy_net.eval()
        self.target_net.eval()
        
        for experience in batch:
            state, act_emb, r_type, reward, next_state, done, next_actions = experience
            
            states.append(state)
            action_embs.append(act_emb)
            relation_onehots.append(self._get_relation_onehot(r_type).squeeze(0).numpy())
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(float(done))
            
            if next_actions and not done:
                best_next_q = -float('inf')
                best_next_emb = np.zeros(self.text_dim)
                best_next_rel = 0
                
                next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    for n_node, n_r_type in next_actions:
                        n_emb = self._encode_node(n_node)
                        n_emb_t = torch.FloatTensor(n_emb).unsqueeze(0).to(self.device)
                        n_rel_onehot_t = self._get_relation_onehot(n_r_type).to(self.device)
                        
                        q_policy = self.policy_net(next_state_t, n_emb_t, n_rel_onehot_t).item()
                        
                        if q_policy > best_next_q:
                            best_next_q = q_policy
                            best_next_emb = n_emb
                            best_next_rel = n_r_type
                
                next_action_embs.append(best_next_emb)
                next_relation_onehots.append(self._get_relation_onehot(best_next_rel).squeeze(0).numpy())
            else:
                next_action_embs.append(np.zeros(self.text_dim))
                next_relation_onehots.append(np.zeros(self.relation_dim))
        
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        action_embs_t = torch.FloatTensor(np.array(action_embs)).to(self.device)
        relation_onehots_t = torch.FloatTensor(np.array(relation_onehots)).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        next_action_embs_t = torch.FloatTensor(np.array(next_action_embs)).to(self.device)
        next_relation_onehots_t = torch.FloatTensor(np.array(next_relation_onehots)).to(self.device)
        weights_t = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        self.policy_net.train()
        q_values = self.policy_net(states_t, action_embs_t, relation_onehots_t)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t, next_action_embs_t, next_relation_onehots_t)
            target_q_values = rewards_t + self.gamma * next_q_values * (1 - dones_t)
        
        td_errors = (q_values - target_q_values).abs().detach().cpu().numpy().flatten()
        self.memory.update_priorities(indices, td_errors)
        
        loss = (weights_t * F.huber_loss(q_values, target_q_values, reduction='none', delta=1.0)).mean()
    
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        self.training_step += 1

        
        return loss.item()
    
    def replay(self) -> float:
        if not self.use_prioritized:
            if len(self.memory) < self.batch_size:
                return 0.0
            batch = random.sample(list(self.memory), self.batch_size)
        return self.replay_prioritized()
    
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filepath: str):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'memory_size': len(self.memory)
        }, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint.get('training_step', 0)
        print(f"✓ Model loaded from {filepath}")
        print(f"  Epsilon: {self.epsilon:.4f}")
        print(f"  Training Step: {self.training_step:,}")

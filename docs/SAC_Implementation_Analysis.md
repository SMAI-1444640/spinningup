# Soft Actor-Critic (SAC) Implementation Analysis

## Spinning Up PyTorch Implementation - Comprehensive Code-Grounded Explanation

---

## Table of Contents
1. [Algorithm Overview](#1-algorithm-overview)
2. [Pseudo-code with Code References](#2-pseudo-code-with-code-references)
3. [End-to-End Data Flow Tracing](#3-end-to-end-data-flow-tracing)
4. [State, Action, and Reward Specification](#4-state-action-and-reward-specification)
5. [Loss Functions and Updates](#5-loss-functions-and-updates)
6. [Implementation Details & Design Choices](#6-implementation-details--design-choices)
7. [Summary Diagram](#7-summary-diagram-textual)

---

## 1. Algorithm Overview

### 1.1 Conceptual Foundation

Soft Actor-Critic (SAC) is an off-policy actor-critic deep reinforcement learning algorithm that optimizes a stochastic policy in an entropy-regularized framework. The key innovation is the **maximum entropy objective**, which augments the standard RL objective with an entropy term:

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]$$

Where:
- $\alpha$ is the temperature parameter controlling the entropy-reward trade-off
- $\mathcal{H}(\pi(\cdot|s_t))$ is the entropy of the policy at state $s_t$

### 1.2 Key Components in This Implementation

| Component | Location | Description |
|-----------|----------|-------------|
| **Entropy Maximization** | [`sac.py:205`](spinup/algos/pytorch/sac/sac.py:205), [`sac.py:230`](spinup/algos/pytorch/sac/sac.py:230) | Entropy bonus subtracted from Q-target and added to policy loss |
| **Twin Q-Networks** | [`core.py:97-98`](spinup/algos/pytorch/sac/core.py:97-98) | `self.q1` and `self.q2` in `MLPActorCritic` |
| **Target Networks** | [`sac.py:165`](spinup/algos/pytorch/sac/sac.py:165) | `ac_targ = deepcopy(ac)` |
| **Temperature (α)** | [`sac.py:50`](spinup/algos/pytorch/sac/sac.py:50) | Fixed hyperparameter `alpha=0.2` |
| **Squashed Gaussian Policy** | [`core.py:29-70`](spinup/algos/pytorch/sac/core.py:29-70) | `SquashedGaussianMLPActor` class |

### 1.3 Version Implemented

**This implementation uses FIXED α (temperature)**, not automatic entropy tuning.

- **Paper Version**: SAC v1 with fixed entropy coefficient
- **Reference**: [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor (arXiv:1801.01290)](https://arxiv.org/abs/1801.01290)
- **Note**: The automatic entropy tuning from SAC v2 ([arXiv:1812.05905](https://arxiv.org/abs/1812.05905)) is NOT implemented

### 1.4 Deviation from Standard Formulation

| Aspect | Standard SAC | This Implementation |
|--------|--------------|---------------------|
| Temperature α | Automatic tuning (SAC v2) | Fixed at 0.2 |
| Target entropy | Computed as -dim(A) | Not used |
| α optimizer | Separate Adam optimizer | None |

---

## 2. Pseudo-code with Code References

### 2.1 High-Level Algorithm

```
Algorithm: Soft Actor-Critic (Fixed α)
═══════════════════════════════════════════════════════════════════════════════

Initialize:
    policy network π_θ (actor)                    → core.py:96 SquashedGaussianMLPActor
    Q-networks Q_φ1, Q_φ2 (critics)               → core.py:97-98 MLPQFunction
    target networks Q_φ'1, Q_φ'2 ← copy(Q_φ1, Q_φ2) → sac.py:165 deepcopy(ac)
    replay buffer D                                → sac.py:175 ReplayBuffer
    temperature α = 0.2                            → sac.py:50 alpha parameter

For each timestep t:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ ENVIRONMENT INTERACTION                                                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ if t < start_steps:                          → sac.py:321-326           │
    │     a ~ Uniform(action_space)                → env.action_space.sample()│
    │ else:                                                                    │
    │     a ~ π_θ(·|s)                             → sac.py:322 get_action(o) │
    │                                                                          │
    │ s', r, done = env.step(a)                    → sac.py:336               │
    │ D.store(s, a, r, s', done)                   → sac.py:352               │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ LEARNING UPDATE (if t >= update_after and t % update_every == 0)        │
    │                                              → sac.py:372-379           │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ For j = 1 to update_every:                   → sac.py:373               │
    │     Sample batch B = {(s,a,r,s',d)} from D   → sac.py:376               │
    │                                                                          │
    │     ┌───────────────────────────────────────────────────────────────────┐
    │     │ Q-FUNCTION UPDATE                      → sac.py:182-219          │
    │     ├───────────────────────────────────────────────────────────────────┤
    │     │ Compute target actions:                                           │
    │     │   ã', log π(ã'|s') = π_θ(s')           → sac.py:192              │
    │     │                                                                    │
    │     │ Compute target Q-values (clipped double-Q):                       │
    │     │   Q_targ = min(Q_φ'1(s',ã'), Q_φ'2(s',ã'))  → sac.py:202-204     │
    │     │                                                                    │
    │     │ Compute Bellman backup:                                           │
    │     │   y = r + γ(1-d)(Q_targ - α·log π(ã'|s'))   → sac.py:205         │
    │     │                                                                    │
    │     │ Update Q-functions:                                               │
    │     │   L_Q = MSE(Q_φ1(s,a), y) + MSE(Q_φ2(s,a), y)  → sac.py:211-213  │
    │     │   φ ← φ - λ_Q ∇_φ L_Q                   → sac.py:246-249         │
    │     └───────────────────────────────────────────────────────────────────┘
    │                                                                          │
    │     ┌───────────────────────────────────────────────────────────────────┐
    │     │ POLICY UPDATE                          → sac.py:222-235          │
    │     ├───────────────────────────────────────────────────────────────────┤
    │     │ Sample actions from current policy:                               │
    │     │   ã, log π(ã|s) = π_θ(s)               → sac.py:224              │
    │     │                                                                    │
    │     │ Compute Q-values (clipped double-Q):                              │
    │     │   Q_π = min(Q_φ1(s,ã), Q_φ2(s,ã))      → sac.py:225-227          │
    │     │                                                                    │
    │     │ Policy loss (maximize Q, maximize entropy):                       │
    │     │   L_π = E[α·log π(ã|s) - Q_π]          → sac.py:230              │
    │     │   θ ← θ - λ_π ∇_θ L_π                  → sac.py:260-263          │
    │     └───────────────────────────────────────────────────────────────────┘
    │                                                                          │
    │     ┌───────────────────────────────────────────────────────────────────┐
    │     │ TARGET NETWORK UPDATE (Polyak averaging)  → sac.py:272-278       │
    │     ├───────────────────────────────────────────────────────────────────┤
    │     │ φ'_i ← ρ·φ'_i + (1-ρ)·φ_i              → sac.py:277-278          │
    │     │ (where ρ = polyak = 0.995)                                        │
    │     └───────────────────────────────────────────────────────────────────┘
    └─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Network Architecture Details

```
Policy Network (SquashedGaussianMLPActor)         → core.py:29-70
═══════════════════════════════════════════════════════════════════════════════
Input: obs [batch, obs_dim]
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Shared MLP Backbone                              → core.py:33               │
│   net = mlp([obs_dim, 256, 256], ReLU, ReLU)                               │
│   Output: [batch, 256]                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ├──────────────────────────────┬──────────────────────────────────────────┐
    ▼                              ▼                                          │
┌────────────────────┐    ┌────────────────────┐                              │
│ mu_layer           │    │ log_std_layer      │   → core.py:34-35           │
│ Linear(256,act_dim)│    │ Linear(256,act_dim)│                              │
│ Output: μ          │    │ Output: log σ      │                              │
│ [batch, act_dim]   │    │ [batch, act_dim]   │                              │
└────────────────────┘    └────────────────────┘                              │
    │                              │                                          │
    │                              ▼                                          │
    │                     ┌────────────────────┐                              │
    │                     │ Clamp log_std      │   → core.py:42              │
    │                     │ [-20, 2]           │                              │
    │                     │ σ = exp(log_std)   │   → core.py:43              │
    │                     └────────────────────┘                              │
    │                              │                                          │
    ▼                              ▼                                          │
┌─────────────────────────────────────────────────────────────────────────────┐
│ Gaussian Distribution: N(μ, σ²)                  → core.py:46               │
│   pi_distribution = Normal(mu, std)                                         │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Reparameterization Trick (rsample)               → core.py:54               │
│   pi_action = μ + σ * ε,  where ε ~ N(0,1)                                 │
│   Output: [batch, act_dim] (unbounded)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Log Probability Computation                      → core.py:62-63            │
│   logp = log N(pi_action | μ, σ²)                                          │
│   logp -= Σ log(1 - tanh²(pi_action))  # Jacobian correction               │
│   Output: [batch,]                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Squashing (tanh) + Scaling                       → core.py:67-68            │
│   pi_action = tanh(pi_action) * act_limit                                   │
│   Output: [batch, act_dim] (bounded to [-act_limit, act_limit])            │
└─────────────────────────────────────────────────────────────────────────────┘


Q-Network (MLPQFunction)                           → core.py:73-81
═══════════════════════════════════════════════════════════════════════════════
Input: obs [batch, obs_dim], act [batch, act_dim]
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Concatenate: [obs, act]                          → core.py:80               │
│   Input: [batch, obs_dim + act_dim]                                         │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ MLP: [obs_dim+act_dim, 256, 256, 1]              → core.py:77               │
│   Activations: ReLU, ReLU, Identity                                         │
│   Output: [batch, 1]                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Squeeze: Remove last dimension                   → core.py:81               │
│   Output: [batch,]                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. End-to-End Data Flow Tracing

### 3.1 Complete Data Path

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA GENERATION (Environment Interaction)                          │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: Get observation from environment
    Location: sac.py:306-308
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ reset_result = env.reset()                                              │
    │ o = reset_result[0] if isinstance(reset_result, tuple) else reset_result│
    │                                                                          │
    │ Data: o ∈ ℝ^obs_dim (numpy array, float32)                              │
    └─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
Step 2: Select action
    Location: sac.py:321-326
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ if t > start_steps:                                                      │
    │     a = get_action(o)  ──────────────────────────────────────────────┐  │
    │ else:                                                                 │  │
    │     a = env.action_space.sample()                                     │  │
    │                                                                       │  │
    │ get_action() flow:                                                    │  │
    │   sac.py:280-282                                                      │  │
    │   ┌─────────────────────────────────────────────────────────────────┐ │  │
    │   │ def get_action(o, deterministic=False):                         │ │  │
    │   │     return ac.act(torch.as_tensor(o, dtype=torch.float32),      │ │  │
    │   │                   deterministic)                                 │ │  │
    │   └─────────────────────────────────────────────────────────────────┘ │  │
    │                                        │                              │  │
    │                                        ▼                              │  │
    │   core.py:100-103                                                     │  │
    │   ┌─────────────────────────────────────────────────────────────────┐ │  │
    │   │ def act(self, obs, deterministic=False):                        │ │  │
    │   │     with torch.no_grad():                                       │ │  │
    │   │         a, _ = self.pi(obs, deterministic, False)               │ │  │
    │   │         return a.numpy()                                        │ │  │
    │   └─────────────────────────────────────────────────────────────────┘ │  │
    │                                                                       │  │
    │ Data: a ∈ ℝ^act_dim (numpy array, float32, bounded by act_limit)     │  │
    └───────────────────────────────────────────────────────────────────────┘  │
                                        │
                                        ▼
Step 3: Execute action in environment
    Location: sac.py:336-342
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ step_result = env.step(a)                                               │
    │ if len(step_result) == 5:                                               │
    │     o2, r, terminated, truncated, _ = step_result                       │
    │     d = terminated or truncated                                         │
    │ else:                                                                    │
    │     o2, r, d, _ = step_result                                           │
    │                                                                          │
    │ Data Generated:                                                          │
    │   o2 ∈ ℝ^obs_dim  (next observation)                                    │
    │   r  ∈ ℝ          (scalar reward)                                       │
    │   d  ∈ {0, 1}     (done flag, modified at sac.py:349)                   │
    └─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: DATA STORAGE (Replay Buffer)                                        │
└─────────────────────────────────────────────────────────────────────────────┘

Step 4: Store transition in replay buffer
    Location: sac.py:352
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ replay_buffer.store(o, a, r, o2, d)                                     │
    │                                                                          │
    │ ReplayBuffer.store() at sac.py:28-35:                                   │
    │ ┌─────────────────────────────────────────────────────────────────────┐ │
    │ │ def store(self, obs, act, rew, next_obs, done):                     │ │
    │ │     self.obs_buf[self.ptr] = obs      # [size, obs_dim]             │ │
    │ │     self.obs2_buf[self.ptr] = next_obs # [size, obs_dim]            │ │
    │ │     self.act_buf[self.ptr] = act      # [size, act_dim]             │ │
    │ │     self.rew_buf[self.ptr] = rew      # [size,]                     │ │
    │ │     self.done_buf[self.ptr] = done    # [size,]                     │ │
    │ │     self.ptr = (self.ptr+1) % self.max_size                         │ │
    │ │     self.size = min(self.size+1, self.max_size)                     │ │
    │ └─────────────────────────────────────────────────────────────────────┘ │
    │                                                                          │
    │ Buffer Structure (initialized at sac.py:20-26):                         │
    │   obs_buf:  np.zeros([replay_size, obs_dim], float32)                   │
    │   obs2_buf: np.zeros([replay_size, obs_dim], float32)                   │
    │   act_buf:  np.zeros([replay_size, act_dim], float32)                   │
    │   rew_buf:  np.zeros([replay_size,], float32)                           │
    │   done_buf: np.zeros([replay_size,], float32)                           │
    └─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: DATA SAMPLING                                                       │
└─────────────────────────────────────────────────────────────────────────────┘

Step 5: Sample batch from replay buffer
    Location: sac.py:376
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ batch = replay_buffer.sample_batch(batch_size)                          │
    │                                                                          │
    │ ReplayBuffer.sample_batch() at sac.py:37-44:                            │
    │ ┌─────────────────────────────────────────────────────────────────────┐ │
    │ │ def sample_batch(self, batch_size=32):                              │ │
    │ │     idxs = np.random.randint(0, self.size, size=batch_size)         │ │
    │ │     batch = dict(                                                   │ │
    │ │         obs=self.obs_buf[idxs],    # [batch_size, obs_dim]          │ │
    │ │         obs2=self.obs2_buf[idxs],  # [batch_size, obs_dim]          │ │
    │ │         act=self.act_buf[idxs],    # [batch_size, act_dim]          │ │
    │ │         rew=self.rew_buf[idxs],    # [batch_size,]                  │ │
    │ │         done=self.done_buf[idxs]   # [batch_size,]                  │ │
    │ │     )                                                               │ │
    │ │     return {k: torch.as_tensor(v, dtype=torch.float32)              │ │
    │ │             for k,v in batch.items()}                               │ │
    │ └─────────────────────────────────────────────────────────────────────┘ │
    │                                                                          │
    │ Output: dict of PyTorch tensors (float32)                               │
    │   batch['obs']:  [batch_size, obs_dim]                                  │
    │   batch['obs2']: [batch_size, obs_dim]                                  │
    │   batch['act']:  [batch_size, act_dim]                                  │
    │   batch['rew']:  [batch_size,]                                          │
    │   batch['done']: [batch_size,]                                          │
    └─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: NETWORK FORWARD PASSES & LOSS COMPUTATION                          │
└─────────────────────────────────────────────────────────────────────────────┘

Step 6: Compute Q-loss (compute_loss_q)
    Location: sac.py:182-219
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ def compute_loss_q(data):                                               │
    │     o, a, r, o2, d = data['obs'], data['act'], data['rew'],            │
    │                      data['obs2'], data['done']                         │
    │                                                                          │
    │     # Current Q-values                                                   │
    │     q1 = ac.q1(o, a)  # [batch_size,]                                   │
    │     q2 = ac.q2(o, a)  # [batch_size,]                                   │
    │                                                                          │
    │     with torch.no_grad():                                               │
    │         # Target actions from current policy                            │
    │         a2, logp_a2 = ac.pi(o2)  # [batch_size, act_dim], [batch_size,]│
    │                                                                          │
    │         # Target Q-values (clipped double-Q)                            │
    │         q1_pi_targ = ac_targ.q1(o2, a2)  # [batch_size,]               │
    │         q2_pi_targ = ac_targ.q2(o2, a2)  # [batch_size,]               │
    │         q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)  # [batch_size,] │
    │                                                                          │
    │         # Bellman backup with entropy bonus                             │
    │         backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)   │
    │         # backup: [batch_size,]                                         │
    │                                                                          │
    │     # MSE loss                                                          │
    │     loss_q1 = ((q1 - backup)**2).mean()  # scalar                       │
    │     loss_q2 = ((q2 - backup)**2).mean()  # scalar                       │
    │     loss_q = loss_q1 + loss_q2           # scalar                       │
    │                                                                          │
    │     return loss_q, q_info                                               │
    └─────────────────────────────────────────────────────────────────────────┘

Step 7: Compute Policy loss (compute_loss_pi)
    Location: sac.py:222-235
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ def compute_loss_pi(data):                                              │
    │     o = data['obs']                                                     │
    │                                                                          │
    │     # Sample actions from current policy (with reparameterization)      │
    │     pi, logp_pi = ac.pi(o)  # [batch_size, act_dim], [batch_size,]     │
    │                                                                          │
    │     # Q-values for sampled actions (clipped double-Q)                   │
    │     q1_pi = ac.q1(o, pi)  # [batch_size,]                              │
    │     q2_pi = ac.q2(o, pi)  # [batch_size,]                              │
    │     q_pi = torch.min(q1_pi, q2_pi)  # [batch_size,]                    │
    │                                                                          │
    │     # Entropy-regularized policy loss                                   │
    │     loss_pi = (alpha * logp_pi - q_pi).mean()  # scalar                │
    │                                                                          │
    │     return loss_pi, pi_info                                             │
    └─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 5: PARAMETER UPDATES                                                   │
└─────────────────────────────────────────────────────────────────────────────┘

Step 8: Update function
    Location: sac.py:244-278
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ def update(data):                                                       │
    │     # ═══════════════════════════════════════════════════════════════  │
    │     # Q-NETWORK UPDATE                                                  │
    │     # ═══════════════════════════════════════════════════════════════  │
    │     q_optimizer.zero_grad()                                             │
    │     loss_q, q_info = compute_loss_q(data)                              │
    │     loss_q.backward()                                                   │
    │     q_optimizer.step()                                                  │
    │                                                                          │
    │     # ═══════════════════════════════════════════════════════════════  │
    │     # FREEZE Q-NETWORKS (efficiency optimization)                       │
    │     # ═══════════════════════════════════════════════════════════════  │
    │     for p in q_params:                                                  │
    │         p.requires_grad = False                                         │
    │                                                                          │
    │     # ═══════════════════════════════════════════════════════════════  │
    │     # POLICY UPDATE                                                     │
    │     # ═══════════════════════════════════════════════════════════════  │
    │     pi_optimizer.zero_grad()                                            │
    │     loss_pi, pi_info = compute_loss_pi(data)                           │
    │     loss_pi.backward()                                                  │
    │     pi_optimizer.step()                                                 │
    │                                                                          │
    │     # ═══════════════════════════════════════════════════════════════  │
    │     # UNFREEZE Q-NETWORKS                                               │
    │     # ═══════════════════════════════════════════════════════════════  │
    │     for p in q_params:                                                  │
    │         p.requires_grad = True                                          │
    │                                                                          │
    │     # ═══════════════════════════════════════════════════════════════  │
    │     # TARGET NETWORK UPDATE (Polyak averaging)                          │
    │     # ═══════════════════════════════════════════════════════════════  │
    │     with torch.no_grad():                                               │
    │         for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):   │
    │             p_targ.data.mul_(polyak)                                    │
    │             p_targ.data.add_((1 - polyak) * p.data)                     │
    └─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. State, Action, and Reward Specification

### 4.1 Tensor Shapes at Runtime

| Tensor | Shape | Type | Location |
|--------|-------|------|----------|
| **Observations (states)** | `[obs_dim]` (single) or `[batch_size, obs_dim]` (batch) | float32 | Extracted at [`sac.py:157`](spinup/algos/pytorch/sac/sac.py:157) |
| **Actions** | `[act_dim]` (single) or `[batch_size, act_dim]` (batch) | float32 | Bounded by `act_limit` at [`sac.py:161`](spinup/algos/pytorch/sac/sac.py:161) |
| **Rewards** | `[]` (scalar) or `[batch_size]` (batch) | float32 | From environment |
| **Done flags** | `[]` (scalar) or `[batch_size]` (batch) | float32 (0.0 or 1.0) | Modified at [`sac.py:349`](spinup/algos/pytorch/sac/sac.py:349) |

### 4.2 Dimension Extraction

```python
# Location: sac.py:157-161
obs_dim = env.observation_space.shape      # e.g., (11,) for Hopper
act_dim = env.action_space.shape[0]        # e.g., 3 for Hopper
act_limit = env.action_space.high[0]       # e.g., 1.0 (assumes symmetric bounds)
```

### 4.3 Action Bounds and Squashing

The implementation uses **tanh squashing** to bound actions:

```python
# Location: core.py:67-68
pi_action = torch.tanh(pi_action)      # Maps to [-1, 1]
pi_action = self.act_limit * pi_action  # Scales to [-act_limit, act_limit]
```

**Critical Assumption** (documented at [`sac.py:160-161`](spinup/algos/pytorch/sac/sac.py:160-161)):
> "Action limit for clamping: critically, assumes all dimensions share the same bound!"

### 4.4 Dimension Propagation Through Networks

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ POLICY NETWORK (SquashedGaussianMLPActor)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ Input:  obs                    [batch_size, obs_dim]                        │
│           │                                                                  │
│           ▼                                                                  │
│ net (MLP backbone)             [batch_size, 256]                            │
│           │                                                                  │
│     ┌─────┴─────┐                                                           │
│     ▼           ▼                                                           │
│ mu_layer    log_std_layer                                                   │
│ [batch, act_dim] [batch, act_dim]                                           │
│     │           │                                                           │
│     └─────┬─────┘                                                           │
│           ▼                                                                  │
│ Normal(mu, std).rsample()      [batch_size, act_dim]  (unbounded)           │
│           │                                                                  │
│           ▼                                                                  │
│ tanh + scale                   [batch_size, act_dim]  (bounded)             │
│           │                                                                  │
│           ▼                                                                  │
│ Output: (action, log_prob)     ([batch, act_dim], [batch,])                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Q-NETWORKS (MLPQFunction)                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ Input:  obs, act               [batch, obs_dim], [batch, act_dim]           │
│           │                                                                  │
│           ▼                                                                  │
│ concat([obs, act])             [batch_size, obs_dim + act_dim]              │
│           │                                                                  │
│           ▼                                                                  │
│ MLP [obs+act, 256, 256, 1]     [batch_size, 1]                              │
│           │                                                                  │
│           ▼                                                                  │
│ squeeze(-1)                    [batch_size,]                                │
│           │                                                                  │
│           ▼                                                                  │
│ Output: Q-value                [batch_size,]                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ TARGET Q-NETWORKS                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ Same architecture as Q-networks                                              │
│ Initialized as: ac_targ = deepcopy(ac)          → sac.py:165                │
│ Frozen: p.requires_grad = False                 → sac.py:168-169            │
│ Updated via Polyak averaging                    → sac.py:272-278            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Loss Functions and Updates

### 5.1 Q-Function Loss

**Mathematical Form:**

$$L_Q(\phi_1, \phi_2) = \mathbb{E}_{(s,a,r,s',d) \sim \mathcal{D}} \left[ \left( Q_{\phi_1}(s,a) - y \right)^2 + \left( Q_{\phi_2}(s,a) - y \right)^2 \right]$$

Where the target $y$ is:

$$y = r + \gamma (1-d) \left( \min_{i=1,2} Q_{\phi'_i}(s', \tilde{a}') - \alpha \log \pi_\theta(\tilde{a}'|s') \right)$$

$$\tilde{a}' \sim \pi_\theta(\cdot|s')$$

**Code Mapping:**

```python
# Location: sac.py:182-219

def compute_loss_q(data):
    o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

    # Q_φ1(s,a) and Q_φ2(s,a)
    q1 = ac.q1(o,a)                                    # Line 185
    q2 = ac.q2(o,a)                                    # Line 186

    with torch.no_grad():
        # ã' ~ π_θ(·|s')
        a2, logp_a2 = ac.pi(o2)                        # Line 192

        # min Q_φ'(s', ã')
        q1_pi_targ = ac_targ.q1(o2, a2)                # Line 202
        q2_pi_targ = ac_targ.q2(o2, a2)                # Line 203
        q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)  # Line 204

        # y = r + γ(1-d)(Q_targ - α·log π)
        backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)  # Line 205

    # L_Q = MSE(Q1, y) + MSE(Q2, y)
    loss_q1 = ((q1 - backup)**2).mean()                # Line 211
    loss_q2 = ((q2 - backup)**2).mean()                # Line 212
    loss_q = loss_q1 + loss_q2                         # Line 213

    return loss_q, q_info
```

### 5.2 Policy Loss

**Mathematical Form:**

$$L_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{D}, \tilde{a} \sim \pi_\theta} \left[ \alpha \log \pi_\theta(\tilde{a}|s) - \min_{i=1,2} Q_{\phi_i}(s, \tilde{a}) \right]$$

**Code Mapping:**

```python
# Location: sac.py:222-235

def compute_loss_pi(data):
    o = data['obs']
    
    # ã ~ π_θ(·|s), log π_θ(ã|s)
    pi, logp_pi = ac.pi(o)                             # Line 224
    
    # min Q_φ(s, ã)
    q1_pi = ac.q1(o, pi)                               # Line 225
    q2_pi = ac.q2(o, pi)                               # Line 226
    q_pi = torch.min(q1_pi, q2_pi)                     # Line 227

    # L_π = E[α·log π - Q]
    loss_pi = (alpha * logp_pi - q_pi).mean()          # Line 230

    return loss_pi, pi_info
```

### 5.3 Temperature Loss (NOT IMPLEMENTED)

This implementation uses **fixed α = 0.2**. The automatic entropy tuning loss would be:

$$L(\alpha) = \mathbb{E}_{a \sim \pi_t} \left[ -\alpha \left( \log \pi_t(a|s) + \bar{\mathcal{H}} \right) \right]$$

Where $\bar{\mathcal{H}}$ is the target entropy (typically $-\dim(\mathcal{A})$).

**This is NOT present in the code** - α is a fixed hyperparameter at [`sac.py:50`](spinup/algos/pytorch/sac/sac.py:50).

### 5.4 Update Sequence

```python
# Location: sac.py:244-278

def update(data):
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Q-NETWORK UPDATE
    # ═══════════════════════════════════════════════════════════════════════
    q_optimizer.zero_grad()                            # Line 246
    loss_q, q_info = compute_loss_q(data)              # Line 247
    loss_q.backward()                                  # Line 248
    q_optimizer.step()                                 # Line 249

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: FREEZE Q-NETWORKS (computational efficiency)
    # ═══════════════════════════════════════════════════════════════════════
    for p in q_params:
        p.requires_grad = False                        # Lines 256-257

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: POLICY UPDATE
    # ═══════════════════════════════════════════════════════════════════════
    pi_optimizer.zero_grad()                           # Line 260
    loss_pi, pi_info = compute_loss_pi(data)           # Line 261
    loss_pi.backward()                                 # Line 262
    pi_optimizer.step()                                # Line 263

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: UNFREEZE Q-NETWORKS
    # ═══════════════════════════════════════════════════════════════════════
    for p in q_params:
        p.requires_grad = True                         # Lines 266-267

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: TARGET NETWORK UPDATE (Polyak averaging)
    # θ_targ ← ρ·θ_targ + (1-ρ)·θ
    # ═══════════════════════════════════════════════════════════════════════
    with torch.no_grad():
        for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
            p_targ.data.mul_(polyak)                   # Line 277
            p_targ.data.add_((1 - polyak) * p.data)    # Line 278
```

---

## 6. Implementation Details & Design Choices

### 6.1 Reparameterization Trick

**Purpose**: Enable gradient flow through stochastic sampling for policy optimization.

**Location**: [`core.py:54`](spinup/algos/pytorch/sac/core.py:54)

```python
pi_action = pi_distribution.rsample()  # Reparameterized sample
```

**Explanation**: Instead of sampling $a \sim \pi_\theta(a|s)$ directly (which blocks gradients), we use:
$$a = \mu_\theta(s) + \sigma_\theta(s) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

This allows gradients to flow through $\mu_\theta$ and $\sigma_\theta$ while $\epsilon$ is treated as a constant.

### 6.2 Log Probability Correction for Tanh Squashing

**Purpose**: Correct the log probability when transforming from unbounded Gaussian to bounded action space.

**Location**: [`core.py:62-63`](spinup/algos/pytorch/sac/core.py:62-63)

```python
logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
```

**Mathematical Derivation** (from SAC paper Appendix C):

For $a = \tanh(u)$ where $u \sim \mathcal{N}(\mu, \sigma^2)$:

$$\log \pi(a|s) = \log \mu(u|s) - \sum_{i=1}^{D} \log(1 - \tanh^2(u_i))$$

The code uses a numerically stable form:
$$\log(1 - \tanh^2(u)) = \log(4) - 2u - 2\cdot\text{softplus}(-2u)$$

### 6.3 Clipped Double-Q Learning

**Purpose**: Combat overestimation bias in Q-learning.

**Locations**: 
- Q-target: [`sac.py:202-204`](spinup/algos/pytorch/sac/sac.py:202-204)
- Policy loss: [`sac.py:225-227`](spinup/algos/pytorch/sac/sac.py:225-227)

```python
# In compute_loss_q (for target)
q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

# In compute_loss_pi (for policy gradient)
q_pi = torch.min(q1_pi, q2_pi)
```

**Explanation**: By taking the minimum of two Q-estimates, we create a pessimistic target that counteracts the tendency to chase overestimated values.

### 6.4 Target Network Smoothing (Polyak Averaging)

**Purpose**: Stabilize training by slowly updating target networks.

**Location**: [`sac.py:272-278`](spinup/algos/pytorch/sac/sac.py:272-278)

```python
with torch.no_grad():
    for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
        p_targ.data.mul_(polyak)                    # ρ·θ_targ
        p_targ.data.add_((1 - polyak) * p.data)     # + (1-ρ)·θ
```

**Default**: `polyak = 0.995` (very slow updates)

### 6.5 Q-Network Freezing During Policy Update

**Purpose**: Computational efficiency - avoid computing Q-network gradients when only updating policy.

**Location**: [`sac.py:254-267`](spinup/algos/pytorch/sac/sac.py:254-267)

```python
# Freeze before policy update
for p in q_params:
    p.requires_grad = False

# ... policy update ...

# Unfreeze after
for p in q_params:
    p.requires_grad = True
```

### 6.6 Log Standard Deviation Clamping

**Purpose**: Prevent numerical instability from extreme variance values.

**Location**: [`core.py:26-27`](spinup/algos/pytorch/sac/core.py:26-27), [`core.py:42`](spinup/algos/pytorch/sac/core.py:42)

```python
LOG_STD_MAX = 2
LOG_STD_MIN = -20

log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
```

**Effect**: Bounds standard deviation to approximately $[2 \times 10^{-9}, 7.4]$.

### 6.7 Initial Random Exploration

**Purpose**: Seed replay buffer with diverse experiences before learning.

**Location**: [`sac.py:321-333`](spinup/algos/pytorch/sac/sac.py:321-333)

```python
if t > start_steps:
    a = get_action(o)
else:
    a = env.action_space.sample()  # Uniform random
```

**Default**: `start_steps = 10000`

### 6.8 Assumptions About Observation/Action Spaces

| Assumption | Location | Implication |
|------------|----------|-------------|
| Continuous action space | [`core.py:90`](spinup/algos/pytorch/sac/core.py:90) | `act_dim = action_space.shape[0]` |
| Symmetric action bounds | [`sac.py:160-161`](spinup/algos/pytorch/sac/sac.py:160-161) | `act_limit = action_space.high[0]` |
| All action dims share same bound | [`sac.py:160-161`](spinup/algos/pytorch/sac/sac.py:160-161) | Single scalar `act_limit` |
| Flat observation space | [`core.py:89`](spinup/algos/pytorch/sac/core.py:89) | `obs_dim = observation_space.shape[0]` |

---

## 7. Summary Diagram (Textual)

### One Full Training Iteration

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        SAC TRAINING ITERATION                                  ║
║                     (when t >= update_after and t % update_every == 0)        ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: SAMPLE BATCH FROM REPLAY BUFFER                                         │
│ Location: sac.py:376                                                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Replay Buffer                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ obs_buf   [1M, obs_dim]  ──┐                                            │   │
│   │ obs2_buf  [1M, obs_dim]  ──┤                                            │   │
│   │ act_buf   [1M, act_dim]  ──┼──▶ Random sample (batch_size=100)          │   │
│   │ rew_buf   [1M,]          ──┤                                            │   │
│   │ done_buf  [1M,]          ──┘                                            │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                           │                                      │
│                                           ▼                                      │
│   batch = {                                                                      │
│       'obs':  [100, obs_dim],  'obs2': [100, obs_dim],                          │
│       'act':  [100, act_dim],  'rew':  [100,],  'done': [100,]                  │
│   }                                                                              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: COMPUTE Q-LOSS                                                          │
│ Location: sac.py:182-219                                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ CURRENT Q-VALUES                                                        │   │
│   │   q1 = ac.q1(obs, act)  ──▶ [100,]                                     │   │
│   │   q2 = ac.q2(obs, act)  ──▶ [100,]                                     │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ TARGET COMPUTATION (no gradients)                                       │   │
│   │                                                                          │   │
│   │   a2, logp_a2 = ac.pi(obs2)           # Sample next actions            │   │
│   │                 ▼                                                        │   │
│   │   q1_targ = ac_targ.q1(obs2, a2)      # Target Q1                       │   │
│   │   q2_targ = ac_targ.q2(obs2, a2)      # Target Q2                       │   │
│   │                 ▼                                                        │   │
│   │   q_targ = min(q1_targ, q2_targ)      # Clipped double-Q                │   │
│   │                 ▼                                                        │   │
│   │   backup = rew + γ(1-done)(q_targ - α·logp_a2)  # Bellman target       │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   loss_q = MSE(q1, backup) + MSE(q2, backup)                                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: UPDATE Q-NETWORKS                                                       │
│ Location: sac.py:246-249                                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   q_optimizer.zero_grad()                                                        │
│   loss_q.backward()           # Compute gradients for q1 and q2                 │
│   q_optimizer.step()          # Update q1 and q2 parameters                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: FREEZE Q-NETWORKS                                                       │
│ Location: sac.py:256-257                                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   for p in q_params:                                                             │
│       p.requires_grad = False   # Don't compute Q gradients during π update    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: COMPUTE POLICY LOSS                                                     │
│ Location: sac.py:222-235                                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ SAMPLE ACTIONS (with reparameterization)                                │   │
│   │   pi, logp_pi = ac.pi(obs)                                              │   │
│   │   # pi: [100, act_dim], logp_pi: [100,]                                 │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ EVALUATE ACTIONS                                                         │   │
│   │   q1_pi = ac.q1(obs, pi)                                                │   │
│   │   q2_pi = ac.q2(obs, pi)                                                │   │
│   │   q_pi = min(q1_pi, q2_pi)   # Clipped double-Q                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   loss_pi = mean(α·logp_pi - q_pi)   # Maximize Q, maximize entropy             │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: UPDATE POLICY NETWORK                                                   │
│ Location: sac.py:260-263                                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   pi_optimizer.zero_grad()                                                       │
│   loss_pi.backward()          # Compute gradients for policy                    │
│   pi_optimizer.step()         # Update policy parameters                        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: UNFREEZE Q-NETWORKS                                                     │
│ Location: sac.py:266-267                                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   for p in q_params:                                                             │
│       p.requires_grad = True    # Re-enable Q gradients for next iteration      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 8: UPDATE TARGET NETWORKS (Polyak Averaging)                               │
│ Location: sac.py:272-278                                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   with torch.no_grad():                                                          │
│       for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):              │
│           p_targ ← 0.995 · p_targ + 0.005 · p                                   │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ Networks Updated:                                                        │   │
│   │   • ac_targ.pi  (target policy - though not used in this implementation)│   │
│   │   • ac_targ.q1  (target Q1)                                             │   │
│   │   • ac_targ.q2  (target Q2)                                             │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════════════════╗
║                           ITERATION COMPLETE                                   ║
║                                                                                ║
║  Parameters Updated:                                                           ║
║    • ac.q1, ac.q2      (via q_optimizer)                                      ║
║    • ac.pi             (via pi_optimizer)                                     ║
║    • ac_targ.q1, ac_targ.q2, ac_targ.pi  (via Polyak averaging)              ║
║                                                                                ║
║  Note: This loop runs `update_every` times per trigger (default: 50)          ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## References

1. **SAC v1**: Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)

2. **SAC v2 (Automatic Temperature)**: Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., ... & Levine, S. (2018). *Soft Actor-Critic Algorithms and Applications*. [arXiv:1812.05905](https://arxiv.org/abs/1812.05905)

3. **Clipped Double-Q**: Fujimoto, S., Hoof, H., & Meger, D. (2018). *Addressing Function Approximation Error in Actor-Critic Methods*. [arXiv:1802.09477](https://arxiv.org/abs/1802.09477) (TD3 paper)

4. **Spinning Up Documentation**: [https://spinningup.openai.com/en/latest/algorithms/sac.html](https://spinningup.openai.com/en/latest/algorithms/sac.html)

---

## File References

| File | Purpose |
|------|---------|
| [`spinup/algos/pytorch/sac/sac.py`](spinup/algos/pytorch/sac/sac.py) | Main SAC algorithm, training loop, loss functions |
| [`spinup/algos/pytorch/sac/core.py`](spinup/algos/pytorch/sac/core.py) | Network architectures (Actor, Critic, ActorCritic) |

---

*Document generated for Spinning Up PyTorch SAC implementation analysis.*

"""GSPO (Group Sequence Policy Optimization).

A group-relative method like GRPO, but the importance ratio is computed at the
sequence level: per-token probabilities are multiplied and length-normalized
(geometric mean) before clipping, instead of clipping per-token ratios. This
demo uses step-level advantage normalization on CartPole as an analogue.
"""
import logging

import copy
import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import rl_utils_gspo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.softmax(self.fc2(out), dim=1)
        return out

# Discounted returns helper
def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

# GSPO算法类
class GSPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, epochs, eps, gamma, beta, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device) # 当前策略网络

        self.ref_model = copy.deepcopy(self.actor) # 参考策略⽹络 (Ref Model)
        self.ref_model.eval()
        self.ref_model.to(device)

        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.gamma = gamma
        self.epochs = epochs
        self.eps = eps
        self.beta = beta
        self.device = device

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        returns = torch.tensor(transition_dict['returns'], dtype=torch.float).view(-1, 1).to(self.device)

        # 新增: 时间步索引
        timesteps = torch.tensor(transition_dict['timesteps'], dtype=torch.long).to(self.device)

        # GSPO 核⼼改进: Step-Level Normalization
        # 我们不直接对所有 returns 做标准化, ⽽是对每⼀个时间步 t 分别做标准化
        # 这样可以消除不同时间步天然的价值差异, 专注于同⼀时刻不同策略选择的优劣

        advantages = torch.zeros_like(returns)
        max_timestep = timesteps.max().item()

        for t in range(max_timestep + 1):
            # 找到当前 batch 中所有处于时间步 t 的样本索引
            idxs = (timesteps == t).nonzero(as_tuple=True)[0]

            if len(idxs) > 1:
                # 提取这些样本的Returns
                t_returns = returns[idxs]

                # 计算该时间步的均值和方差（Baseline）
                mean_t = t_returns.mean()
                std_t = t_returns.std() + 1e-8

                # 计算该时间步的相对优势
                t_advantages = (t_returns - mean_t) / std_t

                # 填回 advantages tensor
                advantages[idxs] = t_advantages
            elif len(idxs) == 1:
                # 如果某一步只有⼀个样本(因为其他trajectory结束了), advantage设为0
                advantages[idxs] = 0.0

        #计算旧概率 (⽤于Ratio)
        with torch.no_grad():
            old_probs = self.actor(states)
            old_log_probs = torch.log(old_probs.gather(1, actions) + 1e-10)

        # 训练循环
        for _ in range(self.epochs):
            probs = self.actor(states)
            log_probs = torch.log(probs.gather(1, actions) + 1e-10)

            # Ratio
            ratio = torch.exp(log_probs - old_log_probs)

            # Loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages
            gspo_loss = torch.min(surr1, surr2).mean()

            # KL penalty
            with torch.no_grad():
                ref_probs = self.ref_model(states)
            all_log_probs = torch.log(probs + 1e-10)
            all_ref_log_probs = torch.log(ref_probs + 1e-10)
            kl_div = torch.sum(probs * (all_log_probs - all_ref_log_probs), dim=1).mean()
            
            # Total Loss
            loss = gspo_loss + self.beta * kl_div

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

if __name__ == "__main__":
    # 配置超参数
    actor_lr = 1e-3
    num_episodes = 600
    batch_size = 10 # Group Size
    hidden_dim = 128
    epochs = 10
    eps = 0.2
    gamma = 0.98
    beta = 0.04 # KL 惩罚系数
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    logger.info("Using device: %s", device)

    env_name = 'CartPole-v1'
    env = gymnasium.make(env_name)
    
    # 设置全局随机种⼦
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 实例化 GSPO Agent
    agent = GSPO(state_dim, hidden_dim, action_dim, actor_lr, 
    epochs, eps, gamma, beta, device)
    
    # 开始训练
    return_list = rl_utils_gspo.train_gspo_agent(env, agent, num_episodes, batch_size)
    
    # --- 画图 ---
    episodes_list = list(range(len(return_list)))
    
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('GSPO on {} (Batch Size={})'.format(env_name, batch_size))
    plt.show()
    plt.savefig('GSPO.png', dpi=300, bbox_inches='tight')

    # 简单的移动平均计算
    def moving_average(a, window_size):
        cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
        return (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) /window_size
    
    mv_return = moving_average(return_list, 9)
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(mv_return)), mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns (Moving Avg)')
    plt.title('GSPO Moving Average')
    plt.show()
    plt.savefig('GSPO_SMOOTH.png', dpi=300, bbox_inches='tight')




"""GRPO (Group Relative Policy Optimization).

A critic-free PPO variant: for each query a group of responses is sampled,
the advantage is the group-normalized reward (no value network / GAE), and the
policy is updated with PPO-style clipping plus a KL penalty against a frozen
reference model to preserve the SFT language ability.

This module contains two parts:
  1. token-level loss primitives (grpo_advantage / grpo_kl / grpo_loss);
  2. a runnable CartPole training loop built on the same objective.
"""
import logging

import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def grpo_advantage(rewards):
    """Group-normalized advantage: (r - mean) / (std + eps)."""
    epsilon = 0.0001
    rewards = torch.tensor(rewards, dtype=torch.float)
    return (rewards - rewards.mean()) / (rewards.std() + epsilon)


def grpo_kl(pi_logprob, pi_ref_logprob):
    """Unbiased KL estimator between current policy and reference."""
    return pi_ref_logprob.exp() / pi_logprob.exp() - (pi_ref_logprob - pi_logprob) - 1

def grpo_loss(pi_logprob,        # 当前策略：每个样本、每个 token 的 log prob
    pi_old_logprob,              # 旧策略：每个样本、每个 token 的 log prob
    pi_ref_logprob,              # 参考策略：每个样本、每个 token 的 log prob
    advantage,                   # 每条回答的 group advantage（每条回答一个值）
    input_len,                   # prompt 长度，用来区分 input 和 response
    len_oi,                      # 每条 response 的长度
    group_num):                  # 一组里有多少条采样结果
    epsilon = 0.2                # PPO clip 的 ε，限制 ratio 不要偏离太多
    beta = 0.01                  # KL 惩罚系数，控制“别离参考模型太远”的强度
    bs, seq_len = pi_logprob.shape
    
    # skip计算采样的每条采样⻓度
    len_oi = torch.tensor([len_oi] * group_num, dtype = torch.long)
    
    # 设定mask, 仅对response为1, 计算loss
    mask = torch.zeros(bs, seq_len)
    mask[:, input_len:] = 1
    
    # GRPO loss
    ratio = torch.exp(pi_logprob - pi_old_logprob)
    ratio_clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    advantage = advantage.unsqueeze(dim = 1) # [a, b ,c] -> [[a], [b], [c]]
    policy_gradient = torch.minimum(ratio * advantage , ratio_clip * advantage)
    
    # 计算KL散度
    kl = grpo_kl(pi_logprob, pi_ref_logprob)
    
    # 严格按照公式计算, 并进⾏mask掩码
    loss = (policy_gradient - beta * kl) * mask
    loss = (-1 / group_num ) * (1/len_oi.unsqueeze(dim = 1)) * loss
    
    # 返回⼀个batch的标量值
    loss = loss.sum()
    
    return loss

if __name__ == '__main__':
    # 输出分布
    pi_logits = torch.randn(3, 5, 32) # batch, seq_len, vocab_size
    pi_ref_logits = torch.randn(3, 5, 32)
    pi_old_logits = torch.randn(3, 5, 32)
    A = grpo_advantage([1,0,1])
    # 获取log prob
    pi_logprob = F.log_softmax(pi_logits, dim = -1)
    pi_ref_logprob = F.log_softmax(pi_ref_logits, dim = -1)
    pi_old_logprob = F.log_softmax(pi_old_logits, dim = -1)
    # group data, 输⼊为11,12,13, 输出模拟采样数据为14,15 ; 15,16 ; 16,17
    token_ids = torch.tensor([[11, 12, 13, 14, 15],
                              [11, 12, 13, 15, 16],
                              [11, 12, 13, 16, 17],])
    
    # 获取policy
    pi_logprob = torch.gather(pi_logprob, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    pi_ref_logprob = torch.gather(pi_ref_logprob, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    pi_old_logprob = torch.gather(pi_old_logprob, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    loss = grpo_loss(pi_logprob, pi_old_logprob, pi_ref_logprob, A, 3, 2, 3)
    logger.info("token-level grpo_loss demo: %s", loss)


# ---------------------------------------------------------------------------
# Runnable CartPole training loop using the GRPO objective.
# ---------------------------------------------------------------------------
import copy

import gymnasium
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import rl_utils


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.softmax(self.fc2(out), dim=1)
        return out
    
# GRPO agent
class GRPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, epochs, eps, gamma, beta, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device) # 当前策略网络
        self.ref_model = copy.deepcopy(self.actor) # 参考策略网络，冻结参数
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
        # 准备数据
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        returns = torch.tensor(transition_dict['returns'], dtype=torch.float).view(-1, 1).to(self.device)

        # GRPO 核⼼步骤 1: Group Relative Advantage
        # 并不需要 Critic 计算 Advantage, ⽽是直接使⽤ Returns
        # 并在"组"的维度上进⾏标准化 (Mean=0, Std=1)
        # 这就是 "Group Relative" 的体现: 在这个 Batch 中, 做得⽐平均好的 advantage > 0
        # transition_dict 中的数据已经是按 Group 组织的
        # 我们假设传⼊的⼀个 batch 就是⼀个 Group 或者多个 Group 的拼接
        # 简单起⻅, 这⾥假设整个 batch 是⼀个⼤的 Group 或者已经被外部归⼀化了
        # 更严谨的做法: 在外部对每个 Group 分别计算 (R - mean)/std, 然后拼接到 returns 中
        # 此处演示的是直接对传⼊的 returns 进⾏标准化 (假设传⼊的是⼀个 Group)
        mean_returns = returns.mean()
        std_returns = returns.std() + 1e-8
        advantages = (returns - mean_returns) / std_returns

        # 计算旧的 log 概率 (Old Log Probs) - ⽤于 GRPO 的 Ratio 计算
        # 注意: 这⾥虽然叫 old_log_probs, 但其实就是数据采集时的概率
        with torch.no_grad():
            old_probs = self.actor(states)
            old_log_probs = torch.log(old_probs.gather(1, actions) + 1e-10)

        # 训练 Epochs
        for _ in range(self.epochs):
            probs = self.actor(states) # 获取当前策略的概率分布
            log_probs = torch.log(probs.gather(1, actions) + 1e-10) # 选定动作的log概率

            # GRPO 核⼼步骤 2: GRPO Ratio
            ratio = torch.exp(log_probs - old_log_probs) # exp(new - old)

            # GRPO Clip Loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages
            grpo_loss = -torch.min(surr1, surr2).mean()

            # GRPO 核心步骤3：KL Divergence Penalty
            # 计算当前策略与 Ref Model 的 KL 散度: D_KL(pi || ref)
            # 公式: sum(p(x) * log(p(x)/q(x)))
            with torch.no_grad():
                ref_probs = self.ref_model(states) # Ref model 输出

            # 为了数值稳定，加上微⼩量
            # KL = sum(probs * (log_probs - log_ref_probs))
            # 注意: 这⾥的 log_probs 是整个分布的 log, 不是 gather 之后的
            all_log_probs = torch.log(probs + 1e-10)
            all_ref_log_probs = torch.log(probs + 1e-10)

            kl_div = torch.sum(probs * (all_log_probs - all_ref_log_probs), dim=1).mean()

            # 总Loss = GRPO Loss + beta * KL
            loss = grpo_loss + self.beta * kl_div

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Training hyper-parameters
actor_lr = 1e-3
num_episodes = 600
batch_size = 10
hidden_dim = 128
epochs = 10
eps = 0.2
gamma = 0.98
beta = 0.04

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env_name = 'CartPole-v1'
# env = gym.make(env_name, render_mode='human')
env = gymnasium.make(env_name)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Instantiate the GRPO agent
agent = GRPO(state_dim, hidden_dim, action_dim, actor_lr, epochs, eps, gamma, beta, device)

# 开始训练
return_list = rl_utils.train_grpo_agent(env, agent, num_episodes, batch_size)

# 画图
episodes_list = list(range(len(return_list)))
plt.figure(figsize=(10, 5))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('GRPO on {} (Batch Size={})'.format(env_name, batch_size))
plt.show()
plt.savefig('GRPO.png', dpi=300, bbox_inches='tight')

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns(Moving Avg)')
plt.title('GRPO on {}'.format(env_name))
plt.show()
plt.savefig('GRPO_SMOOTH.png', dpi=300, bbox_inches='tight')












# GSPO (Group Sequence Policy Optimization) 算法.
    # 和GRPO对比：
        # 两者都属于‘组相对策略优化’
        # 核心差异：Advantage的计算粒度和对齐方式：
            # GRPO (DeepSeek):
                # 序列级/response级A：GRPO里每个 response 的所有 token 共享同一个优势A^,也就是A是序列级的
                # ***token级ratio：虽然公式中写的是序列级别概率连乘，但代码实现时是单个token算ratio，参与clip/loss后再对token聚合，所以实现起来还是属于token级别
                    # 一句话总结：token粒度进入目标
            # GSPO (Alibaba):
                # 序列级A：但是和GRPO不同，GSPO采用不同序列中的同一时间步的reward做组进行组相对优势
                # 序列级ratio：GSPO在公式以及代码实现ratio时是将单个token概率连乘，再clip，并且做长度归一化1/∣y∣，属于序列级处理
                    # 重点：连乘相当于组成序列，然后长度归一化
                    # 一句话总结：先合成一个 sequence 级 ratio，再去 clip 和优化
    # 算法细节：详细见课件pdf
        # 公式：JGSPO​(θ)=Ex∼D, {yi​}i=1G​∼πθold​​(⋅∣x)​[G1​i=1∑G​min(si​(θ)Ai​,clip(si​(θ),1−ϵ,1+ϵ)Ai​)]
            # x∼D：x从训练数据分布D中采样出来
            # {yi​}i=1G​∼πθold​​(⋅∣x)：给定x后，用旧策略πθold采样出G条回答
            # Ai​=r(x,yi​)−mean({r(x,yi​)}i=1G​)​ / std({r(x,yi​)}i=1G​) 
                # 解释：表示第i条回答在同组回答里，相对好多少
            # si​(θ)=(πθ​(yi​∣x) / πθold​​(yi​∣x)​)^1/|yi​|
                # 此处的πθ​(yi​∣x)代表整条序列概率是很多 token 概率连乘：πθ​(yi​∣x)=t=1∏∣yi​∣​πθ​(yi,t​∣x,yi,<t​)
                    # 连乘会带来问题：序列越长，乘出来越容易特别小或特别极端，所以要取下面的几何平均
                # ^1/|yi​|:将连乘的序列级概率分布开序列长度的根号，表示取几何平均
                    # ||：此处不代表绝对值，而是表示序列的长度
    # 代码实现：

'''导入需要的库'''
import copy
# import gym
import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import rl_utils_gspo

'''实现策略网络'''
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.softmax(self.fc2(out), dim=1)
        return out

'''实现GSPO模型'''
# 辅助函数: 计算 Returns
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

'''训练主代码'''
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
    # 针对 Mac ⽤户
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"Using device: {device}")
    
    env_name = 'CartPole-v1'
    # env = gym.make(env_name) # 训练时不渲染以加快速度
    env = gymnasium.make(env_name) # 训练时不渲染以加快速度
    
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




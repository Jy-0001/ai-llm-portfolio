# PPO算法 (Proximal Policy Optimization)
    # 重要性采样(Importance Sampling)
        # 解释：让“旧分布采来的数据”能够近似代表“新分布下的平均效果”的技术
        # 为什么要做重要性采样：
            # 采样成本高昂：不用的话，更新策略后旧数据旧过时了，需要重新采样，成本高
        # 核心公式：Ex∼p​[f(x)]=Ex∼q​[{p(x) / q(x)} *​f(x)]
            # p(x)：目标分布（真正想要的那个分布），旧策略动作条件概率πold(a|s)
            # q(x)：采样分布（手里实际采样得到数据时用的分布），新策略动作条件概率πθ(a|s)
            # PPO核心比值：p(x) / q(x) 相当于：πθ​(a∣s) / πold​(a∣s)
        # 核心比值(ratio)：表示新策略更新了多少（步长）
            # 公式：rt​(θ)=πθ​(at​∣st​)​ / πθold​​(at​∣st​)
            # 解释：用权重告诉你：旧数据里的这条样本，在新策略世界里应该算重一点还是轻一点
            # 具体解释：新策略动作条件概率除以旧策略动作条件概率。对于动作a，新策略概率较高，旧策略概率较低，则比值>1，则给更大权重，反之比值<1,给更小的权重。
        # 优势：可多轮复用，因为概率比值相对于直接更新策略可以在新策略和旧策略离得不远的时候多次比较
        # 注意：重要性采样需要同时有“旧策略分布”和“当前策略分布”做比较；只有当当前参数相对旧参数发生变化时，这个比值才不再是 1，重要性采样才真正起作用。
    # 算法原理：
        # PPO-penalty惩罚
            # 解释：相对于TRPO的s.t，ppo penalty用拉格朗日乘数法直接将KL散度的限制放进了目标函数中
            # 公式：max​Es∼νk​,a∼πθk​​(a∣s)​[rt​(θ)Aπθk​​(s,a)−βDKL​(πθk​​(⋅∣s)∥πθ​(⋅∣s))]
                # rt​(θ) = πθ(a|s) / πθk(a|s)
                # 其中d_k​=D_KL​，意味实际KL距离，也就是更新前后的概率变化大小
                # β为惩罚强度，随KL距离动态变化，β越大惩罚越大，更新会越保守，β越小惩罚越小，更新会越激进
                # δ为目标阈值，表示希望KL大概控制在这个范围
                    # 更新时为什么要进行1.5的乘除？：因为要给目标KL一个容忍区间，避免β因为小的波动频繁改动，1.5为经验取值
                # β的更新规则：
                    # 如果d_k < δ/1.5：则下一步的β变化为：βk+1 = βk / 2 (实际KL距离较小，鼓励更大步更新，β缩小一半)
                    # 如果d_k > δ*1.5：则下一步的β变化为：βk+1 = βk * 2 (实际KL距离过大，惩罚大步更新，β变为2倍)
            # 缺点：β需要动态调整，比较麻烦  
        # PPO-clip截断
            # 解释：ppo-clip更加直接的把ratio(核心比值)限制在区间之内
            # 公式：max_θ​Es∼νk​,a∼πθk​​(a∣s)​[min(rt​(θ)Aπθk​​(s,a),clip(rt​(θ),1−ϵ,1+ϵ)Aπθk​​(s,a))]
                # rt​(θ) = πθ(a|s) / πθk(a|s)
                # clip(x,l,r) 等同于 max(min(x,r),l)，即把x限制在[l,r]内
                # ϵ为超参数，表示进行截断的范围
                # min的作用：在原始目标rt​(θ)Aπθk​​(s,a)和截断目标clip(rt​(θ),1−ϵ,1+ϵ)Aπθk​​(s,a)里选更保守的那一个
                # clip过程：
                    # 如果Aπθk​​(s,a) > 0说明这个动作的价值⾼于平均值, 最大化这个式子会增大ratio, 但不会让其超过1+ϵ
                    # 如果Aπθk​​(s,a) < 0说明这个动作的价值低于平均值, 最大化这个式子会减小ratio, 但不会让其小于1−ϵ
            # 优点：# 不用调β，实现更简单，实践中更常用
        # 注意：PPO-penalty和PPO-clip不是必须同时存在，而是PPO两种版本
    # 背后的思路：
        # 回顾：策略梯度：
            # 公式：∇J(θ)=E[∇logπ(a∣s)⋅G]
                # ∇logπ(a∣s)为参数θ调整的方向
                # G为回报，是代表符号。
        # 优势函数：策略梯度引入baseline：表示这个动作好不好（方向）
            # ∇J(θ)=E[∇logπ(a∣s)⋅(Q(s,a) - V(s))]
                # Q(s,a)为做动作a的价值
                # V(s)为这个状态本身的平均水平(baseline)，只依赖状态s，不依赖动作a
                # A(s,a) = Q(s,a) - V(s)，即优势函数
            # 好处：
                # 数学上能降低方差，让参数变化更稳定，但不改变期望
    # 代码实现：

# ⼤量实验表明, PPO-clip总是⽐PPO-penalty表现得更好, 因此下⾯专注于PPO-clip的代码实现
import torch 
import torch.nn as nn
import torch.nn.functional as F
# import gym
import gymnasium
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import rl_utils

'''策略梯度⽹络类的定义'''
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.softmax(self.fc2(out), dim=1)
        return out

'''价值梯度⽹络类的定义'''
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, 1) # 价值⽹络输⼊是⼀个状态, 输出是状态的价值, ⼀个单⼀的浮点数值, 所有输出通道是1
    
    def forward(self, x):
        out = F.relu(self.fc1(x))
        
        return self.fc2(out)
    
'''实现PPO模型'''
class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda

        self.epochs = epochs

        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)

        probs = self.actor(state)

        action_distribution = torch.distributions.Categorical(probs)

        action = action_distribution.sample()

        return action.item()
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)

        td_error = td_target - self.critic(states)

        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_error.cpu()).to(self.device)

        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            s1 = ratio * advantage
            s2 = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) * advantage

            actor_loss = torch.mean(-torch.min(s1,s2))

            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()

'''实现训练代码'''
actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 500
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env_name = 'CartPole-v1'
# env = gym.make(env_name, render_mode='human')
env = gymnasium.make(env_name)

random.seed(0)
np.random.seed(0)
env.reset(seed=0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, 
lmbda, epochs, eps, gamma, device)
return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

'''下⾯代码⽤来画图'''
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()
plt.savefig('PPO.png', dpi=300, bbox_inches='tight')

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()
plt.savefig('PPO_SMOOTH.png', dpi=300, bbox_inches='tight')








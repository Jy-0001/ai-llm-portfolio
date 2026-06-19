# DPO算法 (Direct Preference Optimization)
    # 解释：通过直接优化语言模型，避免了传统RLHF中的奖励模型构建和强化学习步骤, 以更好的满⾜⼈类偏好. DPO利⽤⼆元交叉熵作为⽬标函数, 简化了偏好学习过程并提⾼了效率.
    # 原理：
        # 回顾：标准的RLHF流程
            # SFT (监督微调): ⽤⾼质量数据微调基础模型
            # 奖励模型训练: 收集⼈类偏好数据 (A vs B⽐较), 训练奖励模型
            # RL优化: ⽤PPO等算法优化策略, 最⼤化奖励同时控制偏离SFT模型
        # 和RLHF对比：
            # 输入：DPO的输入不是RL四元组，而是偏好对：(x,yw​,yl​)
            # DPO的目标不是学V或Q，而是直接学策略πθ
            # DPO 不显式训练 reward model，但它背后有“隐式 reward”解释
            # DPO 一直拿参考模型 πref当锚点，防止当前模型漂太远
        # 推导过程
            # 理论基础：Bardley-Terry模型：
                # 公式：P(y1​≻y2​∣x)=σ(r(x,y1​)−r(x,y2​))
                # 含义：偏好=reward差值：如果r(x,y1​) > r(x,y2​)，那么y1的结果更符合人类偏好
            # RLHF 告诉我们：最优策略和 reward 有固定关系：
                # π^∗(y∣x)∝π_ref​(y∣x)⋅e^(r(x,y)/β)
                # 含义：reward 越高 → 概率越大：最优策略 = 参考模型概率 * reward指数权重
            # 所以 reward 可以反过来写成策略的函数：
                # r(x,y)=βlog(πref​(y∣x) / π∗(y∣x))​+βlogZ(x)
                # reward ≈ “当前策略相对参考策略的偏好程度”
            # 代回偏好模型后，reward 被消掉
                # P(y1​≻y2​∣x)=σ(βlog(π_ref​(y1​∣x) / π_ref(y1​∣x) ​− βlog(π_ref​(y2​∣x) / π(y2​∣x))​
                # 关键过程：Z(x) 被消掉
            # 最后得到一个直接训练策略的损失函数：
                # L_DPO​=−E_(x,yw​,yl​)∼D​[logσ(βlog(πθ​(yw​∣x) / πref​(yw​∣x))​−βlog(πθ​(yl​∣x) / πref​(yl​∣x))​)]
                    # yw:人类喜欢
                    # yl:人类不喜欢
                    # log(πθ​(y∣x) / πref​(y∣x)​)：当前模型相对参考模型，对这个回答更偏爱多少
                    # D:数据集
    # 代码实现：

import torch 
import torch.nn as nn
import torch.nn.functional as F
# import gym
import gymnasium
import random
import copy
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
    
'''DPO算法类'''
class DPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, beta, epochs, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)

        self.ref_model = copy.deepcopy(self.actor)
        self.ref_model.eval()
        self.ref_model.to(device)

        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.beta = beta
        self.epochs = epochs
        self.device = device

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)

        action_distribution = torch.distributions.Categorical(probs)
        action = action_distribution.sample()

        return action.item()
    
    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        action_w = torch.tensor(transition_dict['actions_w']).view(-1, 1).to(self.device)
        action_l = torch.tensor(transition_dict['actions_l']).view(-1, 1).to(self.device)

        for _ in range(self.epochs):
            probs = self.actor(states)
            log_probs = torch.log(probs + 1e-10) # 数值稳定性:避免log(0)产生NaN的loss

            policy_log_prob_w = log_probs.gather(1,action_w)
            policy_log_prob_l = log_probs.gather(1,action_l)

            with torch.no_grad():
                ref_probs = self.ref_model(states)
                ref_log_probs = torch.log(ref_probs + 1e-10)
                ref_log_prob_w = ref_log_probs.gather(1, action_w)
                ref_log_prob_l = ref_log_probs.gather(1, action_l)
            
            logits_w = policy_log_prob_w - ref_log_prob_w
            logits_l = policy_log_prob_l - ref_log_prob_l
            logits_diff = logits_w - logits_l

            loss = -F.logsigmoid(self.beta * logits_diff).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

'''完成针对DPO训练的⼯具代码函数'''
def train_dpo_online_agent(env, agent, num_episodes):
    return_list = []
    action_dim = env.action_space.n

    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {
                    'states':[], 
                    'actions_w':[], # 胜者动作 (实际采取的动作)
                    'actions_l':[], # 败者动作 (未采取的动作)
                    'next_states':[], 
                    'dones':[]
                }

                state, info = env.reset()
                done = False

                while not done:
                    action = agent.take_action(state)

                    action_l = action
                    while action_l == action:
                        action_l = random.randint(0, action_dim - 1)

                    next_state, reward, terminated, truncated, _ =env.step(action)
                    done = terminated or truncated

                    transition_dict['states'].append(state)
                    transition_dict['actions_w'].append(action)
                    transition_dict['actions_l'].append(action_l)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['dones'].append(done)

                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)

                agent.update(transition_dict)

                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

    return return_list


'''训练代码的编写'''
# 参数设置
actor_lr = 1e-3
num_episodes = 500
hidden_dim = 128
epochs = 10
beta = 0.5 # DPO 对于 beta ⽐较敏感, 建议设⼤⼀点以保持稳定性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env_name = 'CartPole-v1'
# env = gym.make(env_name, render_mode='human')
env = gymnasium.make(env_name)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DPO(state_dim, hidden_dim, action_dim, actor_lr, beta, epochs, device)

return_list = train_dpo_online_agent(env, agent, num_episodes)

# episodes_list = list(range(len(return_list)))

'''下⾯代码⽤来画图'''
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DPO on {}'.format(env_name))
plt.show()
plt.savefig('DPO.png', dpi=300, bbox_inches='tight')

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()
plt.savefig('DPO_SMOOTH.png', dpi=300, bbox_inches='tight')





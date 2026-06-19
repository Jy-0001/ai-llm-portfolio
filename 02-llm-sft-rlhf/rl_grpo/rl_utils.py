import numpy as np
import torch
import collections
import random
from tqdm import tqdm


# 计算累计折扣回报 (Monte Carlo Returns)
# 这是 GRPO 替代 Critic 的关键：直接用真实回报作为基准
def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


# --- GRPO 专用的训练函数 ---
def train_grpo_agent(env, agent, num_episodes, batch_size=5):
    """
    batch_size: 每次更新前采集的 Episode 数量 (构成一个 Group)
    """
    return_list = []
    
    # 迭代次数 = 总局数 / 每组局数
    num_iterations = int(num_episodes / batch_size)
    
    with tqdm(total=num_iterations, desc='GRPO Training') as pbar:
        for i in range(num_iterations):
            # 1: 采集一组(Group)数据
            # GRPO 的核心是基于一组采样来计算相对优势
            batch_transition_dict = {
                'states': [], 
                'actions': [], 
                'returns': [],   # 存储计算好的折扣回报
                'dones': []
            }
            
            group_rewards = []
            
            # 在这一个 Batch (Group) 中采集多条链
            for _ in range(batch_size):
                state, info = env.reset()
                done = False
                episode_rewards = []
                episode_states = []
                episode_actions = []
                episode_dones = []
                
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    episode_states.append(state)
                    episode_actions.append(action)
                    episode_rewards.append(reward)
                    episode_dones.append(done)
                    
                    state = next_state
                
                # 记录本局总分
                total_reward = sum(episode_rewards)
                return_list.append(total_reward)
                group_rewards.append(total_reward)
                
                # 计算本局的 Monte Carlo 回报 (Returns)
                # GRPO 没有 Critic, 直接用 MC Return 加上标准化作为 Advantage
                returns = compute_returns(episode_rewards, agent.gamma)
                
                # 将数据存入 batch 字典
                batch_transition_dict['states'].extend(episode_states)
                batch_transition_dict['actions'].extend(episode_actions)
                batch_transition_dict['returns'].extend(returns)
                batch_transition_dict['dones'].extend(episode_dones)

            # 2: 更新策略
            # 将这一组数据传给 agent 进行 GRPO 更新
            agent.update(batch_transition_dict)
            
            # 3: 打印进度
            if (i + 1) % 1 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (batch_size * (i + 1)), 
                    'avg_return': '%.2f' % np.mean(group_rewards)
                })
            pbar.update(1)
            
    return return_list


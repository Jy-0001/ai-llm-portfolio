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


# --- GSPO 专用的训练函数 ---
def train_gspo_agent(env, agent, num_episodes, batch_size=5):
    '''
    GSPO 训练函数
    关键改变: 为了实现 Group Step-Level 对⽐, 同⼀个 Batch (Group) 内的 Episode
    必须拥有相同的初始状态 (通过固定 Seed 实现),这样才满足独立同分布
    '''
    return_list = []
    num_iterations = int(num_episodes / batch_size)
    
    with tqdm(total=num_iterations, desc='GSPO Training') as pbar:
        for i in range(num_iterations):
            batch_transition_dict = {
                'states': [], 
                'actions': [], 
                'returns': [], 
                'timesteps': [], # 新增: 记录时间步
                'dones': [],
                'ep_ids': []
                }
            group_rewards = []

            # 关键点: 同状态采样 (Same-State Sampling)
            # 随机⽣成⼀个种⼦, ⽤于这⼀组的初始化, 这样这 batch_size 个 episode 都会⾯对完全相同的初始情况, 从⽽使得 Step-level 的⽐较是公平的
            group_seed = random.randint(0, 100000)

            for id in range(batch_size):
                # 使用相同的seed重置环境
                state, info = env.reset(seed=group_seed)

                done = False
                episode_rewards = []
                episode_states = []
                episode_actions = []
                episode_timesteps = []
                episode_id = []
                step_count = 0
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    episode_states.append(state)
                    episode_actions.append(action)
                    episode_rewards.append(reward)
                    episode_timesteps.append(step_count)
                    episode_id.append(id)
                    
                    state = next_state
                    step_count += 1

                # 记录回报
                total_reward = sum(episode_rewards)
                return_list.append(total_reward)
                group_rewards.append(total_reward)

                # 计算 Monte Carlo Returns
                returns = compute_returns(episode_rewards, agent.gamma)

                # 数据采样
                batch_transition_dict['states'].extend(episode_states)
                batch_transition_dict['actions'].extend(episode_actions)
                batch_transition_dict['returns'].extend(returns)
                batch_transition_dict['timesteps'].extend(episode_timesteps)
                batch_transition_dict['dones'].extend([False]*len(episode_rewards))
                batch_transition_dict['ep_ids'].extend(episode_id)

                # 注意: CartPole的reset seed只在reset时⽣效, step后随机性由action决定
                # 这种⽅式保证了t=0的状态完全⼀致, 后续状态虽然因action不同⽽发散, 但GSPO⽐较的就是在发散路径上谁做得更好
                
                # 更新策略
                agent.update(batch_transition_dict)
                # 打印与记录
                if (i + 1) % 1 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (batch_size * (i + 1)), 
                        'avg_return': '%.2f' % np.mean(group_rewards)
                    })
                pbar.update(1)
    
    return return_list
                
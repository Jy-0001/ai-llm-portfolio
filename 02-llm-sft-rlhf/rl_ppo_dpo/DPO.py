"""DPO (Direct Preference Optimization).

Optimizes the policy directly from preference pairs (x, y_win, y_lose) with a
binary cross-entropy objective, skipping the explicit reward model and RL loop
of classic RLHF. A frozen reference model anchors the policy to prevent drift.
Loss: -log sigmoid(beta * (logratio_win - logratio_lose)).
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import rl_utils

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
    
# DPO agent
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

# Online preference-pair sampling and training loop
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


# Training hyper-parameters
actor_lr = 1e-3
num_episodes = 500
hidden_dim = 128
epochs = 10
beta = 0.5  # DPO is sensitive to beta; a larger value keeps training stable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env_name = 'CartPole-v1'
env = gymnasium.make(env_name)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DPO(state_dim, hidden_dim, action_dim, actor_lr, beta, epochs, device)

return_list = train_dpo_online_agent(env, agent, num_episodes)

# Plot training returns
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





# SAPO算法(Soft Adaptive Policy Optimization)
    # 解释：在grpo的基础上，摒弃clip，将重要性采样ratio送进平滑函数fi,t()，结果与A相乘，再对token取平均、对组取平均
    # 特点：引⼊⼀个平滑, 温度可控的软⻔控机制, 替代硬截断（clip）:
        # 当某个token的重要性⽐接近 1 (即 "on-policy") 时, 梯度完整保留.
        # 当偏离增⼤时, 梯度被连续衰减⽽⾮突变为零, 从⽽构建⼀个 "软信任域".
    # 算法细节：详见课件pdf
        # 公式：J(θ)=Eq∼D,{yi​}i=1G​∼πθold​​(⋅∣q) {1/G * ​i=1∑G 1/|yi|1​t=1∑|yi|​fi,t​(ri,t​(θ))Ai}​
            # soft gate：​fi,t​(x) = σ(τi,t​(x−1)) * 4 / τi,t​
                # σ(⋅):sigmoid函数：σ(x)=1+e−x1​
                    # 特点：
                        # 输入很小 → 输出接近 0
                        # 输入适中 → 输出平滑变化
                        # 输入很大 → 输出接近 1
                # x-1:因为ri,t​(θ)(ratio)中心点是1
                # τi,t：温度参数
                    # 公式：τi,t = τpos​,​if A^i​>0
                    #              τneg​,otherwise
                        # 含义：如果这条回答A是正的，用τpos。如果这条回答A是负的，用τneg
                # 4：因为x=0时σ函数导数为0.25，为了抵消而乘一个4
            # Ai(实际为A尖i)：GRPO公式同款序列级粒度优势函数A，计算序列组内平均回报，并非序列间组
            # ri,t​(θ)：GRPO公式同款ratio，表示第i条回答第t个token的重要性采样比率
                # ri,t = 1：没变
                # ri,t > 1：新策略更偏向这个token
                # ri,t < 1：新策略更不偏向这个token
        # 公式求导后的梯度形式：
            # 公式：∇θJ(θ)=Eq∼D,{yi}i=1G∼πθold(⋅|q) [ 1/G * i=1∑G 1/|yi| * t=1∑|yi| wi,t(θ) * ri,t(θ) * A^i * ∇θ log πθ(yi,t | q, yi,<t) ]
                # 含义：目标函数对参数θ求导后，真正更新参数时看的不是fi,t()本身，而是每个token对应的梯度项
                # 结构拆解：
                    # wi,t(θ)：soft gate求导后自然产生的权重项，决定这个token的梯度保留多少
                    # ri,t(θ)：第i条回答第t个token的重要性采样比率
                    # A^i：第i条回答的序列级优势，表示这条回答整体好不好
                    # ∇θ log πθ(yi,t | q, yi,<t)：第i条回答第t个token的log概率对参数θ的梯度，是真正推动参数更新的部分
                # 直觉：SAPO最终是在“原始policy gradient”的前面，又乘了一个平滑权重wi,t(θ)
                    # 如果某个token比较正常(on-policy)，这个权重大，梯度保留
                    # 如果某个token偏得很远(off-policy)，这个权重变小，梯度被衰减
            # 为什么会多出wi,t(θ)：
                # 因为目标函数里不是直接写ri,t(θ)，而是写fi,t(ri,t(θ)))
                # 对θ求导时要用链式法则：
                    # ∂/∂θ fi,t(ri,t(θ)) = fi,t'(ri,t(θ)) * ∂ri,t(θ)/∂θ
                # 其中：
                    # fi,t'(ri,t(θ)) 就变成了后面的 wi,t(θ)
                    # ∂ri,t(θ)/∂θ 会变成 ri,t(θ) * ∇θ log πθ(yi,t | q, yi,<t)
            # ri,t(θ)为什么求导后会变成 ri,t(θ) * ∇θ log πθ(...)：
                # 因为 ratio = πθ / πθold
                # 分母πθold与当前参数θ无关，所以只对分子求导
                # 利用恒等式：∇θ πθ = πθ * ∇θ log πθ
                # 因此：∇θ ri,t(θ) = ri,t(θ) * ∇θ log πθ(yi,t | q, yi,<t)
            # wi,t(θ)的公式：
                # 公式：wi,t(θ) = 4 * pi,t(θ) * (1 - pi,t(θ))
                # 其中：pi,t(θ) = σ(τi,t(ri,t(θ)-1))
                # 含义：wi,t(θ)本质上就是sigmoid导数形状对应的权重项
                    # 当ri,t(θ)接近1时，pi,t≈0.5，wi,t最大
                    # 当ri,t(θ)远离1时，pi,t趋近0或1，wi,t变小
            # wi,t(θ)的性质：
                # 当ri,t(θ)=1时：
                    # pi,t(θ)=σ(0)=0.5
                    # wi,t(θ)=4*0.5*(1-0.5)=1
                    # 含义：当token正好on-policy时，梯度完整保留，不缩放
                # 当ri,t(θ)偏离1越来越远时：
                    # wi,t(θ)会平滑下降到接近0
                    # 含义：不是像clip一样直接截断，而是连续衰减
            # 所以SAPO的核心：
                # 不是“超过边界就把梯度砍掉”
                # 而是“根据每个token偏离1的程度，给这个token的梯度乘一个连续变化的权重”
                # 这就是token级别的软门控
            # 和GRPO/GSPO的区别：
                # GRPO：token级ratio + hard clip，超界后梯度直接被截断
                # GSPO：sequence级ratio，整条序列共用一个ratio
                # SAPO：token级ratio + soft gate，只对偏离严重的token逐个衰减，不连坐整条序列
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
import rl_utils_sapo

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
    
'''SAPO算法'''
class SAPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr,epochs, gamma, beta, tau_pos, tau_neg, device):
        # tau_pos: 正优势样本的温度系数 (例如 1.0)
        # tau_neg: 负优势样本的温度系数 (通常 > tau_pos, 例如10.0, 以快速衰减错误⽅向的梯度)
        # 当前策略⽹络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)

        # 参考策略⽹络（用于KL散度约束）
        self.ref_model = copy.deepcopy(self.actor)
        self.ref_model.eval()
        self.ref_model.to(device)

        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.gamma = gamma
        self.epochs = epochs # 每次 update 内部训练多少轮
        self.beta = beta # KL 惩罚系数
        self.tau_pos = tau_pos # SAPO 核⼼参数: 正向温度
        self.tau_neg = tau_neg # SAPO 核⼼参数: 负向温度
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

        # 步骤 1: Group Relative Advantage(组相对优势), 这⾥是对整个Group(Batch)进⾏标准化
        mean_returns = returns.mean()
        std_returns = returns.std() + 1e-8
        advantages = (returns - mean_returns) / std_returns

        # 计算旧的 log 概率(Old Log Probs) - ⽤于计算 Ratio
        with torch.no_grad():
            old_probs = self.actor(states)
            old_log_probs = torch.log(old_probs.gather(1, actions) + 1e-10)
        
        # 训练 Epochs
        for _ in range(self.epochs):
            # 获取当前策略的概率分布
            probs = self.actor(states)
            log_probs = torch.log(probs.gather(1, actions) + 1e-10)

            '''SAPO核心实现'''
            # 计算重要性采样⽐率 r_t(\theta)
            # ratio = exp(new_log - old_log)
            ratio = torch.exp(log_probs - old_log_probs)

            # 确定⾮对称温度 tau (Asymmetric Temperature)
            # 论⽂中: if A > 0 then tau_pos, else tau_neg
            # 形状需要与 advantages ⼀致
            tau = torch.where(advantages > 0, 
                              torch.tensor(self.tau_pos, device=self.device), 
                              torch.tensor(self.tau_neg, device=self.device))
            
            # 计算软⻔控函数 f(r) (Soft Gate)
            # 论⽂公式 (6): f(x) = sigmoid(tau * (x - 1)) * (4 / tau)
            # 注意: x 就是 ratio
            soft_gate = torch.sigmoid(tau * (ratio - 1)) * (4.0 / tau)

            # 计算 SAPO Loss, J = E [ f(r) * A ], Loss = -J
            sapo_loss = -(soft_gate * advantages).mean() # 此处mean()操作直接完成组平均和token平均（1/G和1/|yi|）
            
            # 辅助 Loss: KL Divergence Penalty (可选但推荐)
            # 虽然SAPO⾃带软信任域, 但保留KL惩罚可防⽌策略过拟合Reward或偏离基座模型太远
            with torch.no_grad():
                ref_probs = self.ref_model(states)

            all_log_probs = torch.log(probs + 1e-10)
            all_ref_log_probs = torch.log(ref_probs + 1e-10)

            # KL(pi || ref)
            kl_div = torch.sum(probs * (all_log_probs - all_ref_log_probs), dim=1).mean()
            
            # 总 Loss
            loss = -sapo_loss + self.beta * kl_div

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

'''训练主代码'''
if __name__ == '__main__':
    # --- 超参数配置 ---
    actor_lr = 1e-3
    num_episodes = 800 # 总训练局数
    batch_size = 10 # Group Size (每组样本数)
    hidden_dim = 128
    epochs = 10 # 每次更新的内循环次数
    gamma = 0.98 # 折扣因⼦
    beta = 0.01 # KL 惩罚系数 (SAPO ⾃带软约束，beta 可以较⼩)
    
    # --- SAPO 特有参数 ---
    # 论⽂建议 tau_neg > tau_pos
    # tau_pos: 控制正样本的梯度衰减 (类似于 PPO clip 的作⽤，值越⼤，允许偏离越⼩)
    # tau_neg: 强烈抑制表现差的 Off-policy 样本
    tau_pos = 1.0
    tau_neg = 10.0

    # 设备检测
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    print(f'Using device:{device}')

    # 环境设置
    env_name = 'CartPole-v1'
    # env = gym.make(env_name) # 训练不渲染
    env = gymnasium.make(env_name) # 训练不渲染

    # 设置随机种⼦
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 实例化 SAPO Agent
    agent = SAPO(state_dim, hidden_dim, action_dim, actor_lr, 
        epochs, gamma, beta, 
        tau_pos, tau_neg, device
        )
    
    # 开始训练
    return_list = rl_utils_sapo.train_sapo_agent(env, agent, num_episodes, batch_size)

    # --- 结果可视化 ---

    # 1. 原始 Return 曲线
    episodes_list = list(range(len(return_list)))
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_list, return_list, alpha=0.5, label='Raw Returns')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'SAPO on {env_name} (tau_pos={tau_pos}, tau_neg={tau_neg})')
    plt.show()
    plt.savefig('SAPO.png', dpi=300, bbox_inches='tight')

    # 2. 移动平均曲线
    def moving_average(a, window_size):
        cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
        return (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    
    window_size = 20
    if len(return_list) > window_size:
        mv_return = moving_average(return_list, window_size)
    plt.plot(range(window_size-1, len(return_list)), mv_return, color='red', label='Moving Avg')
    
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('SAPO_SMOOTH.png', dpi=300, bbox_inches='tight')



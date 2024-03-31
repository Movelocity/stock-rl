
import numpy as np
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from qnetwork import QNetwork


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device, seed=0):
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple( # 临时结构体
            "Experience", 
            # field_names=["state", "feature", "action", "reward", "next_state", "next_feature", "done"]
            field_names=["state", "action", "reward", "next_state", "done"]
        )
        self.seed = random.seed(seed)
        self.device = device
    
    # def add(self, state, feature, action, reward, next_state, next_feature, done):
    #     """增加一条记录"""
    #     e = self.experience(state, feature, action, reward, next_state, next_feature, done)
    #     self.memory.append(e)
    def add(self, state, action, reward, next_state, done):
        """增加一条记录"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, device=None):
        """从记忆库中随机采样一个批次的记录"""
        if device is None:
            device = self.device
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.FloatTensor(np.stack([e.state for e in experiences if e is not None], axis=0)).to(device)
        actions = torch.LongTensor(np.vstack([e.action for e in experiences if e is not None])).to(device)
        rewards = torch.FloatTensor(np.vstack([e.reward for e in experiences if e is not None])).to(device)
        next_states = torch.FloatTensor(np.stack([e.next_state for e in experiences if e is not None], axis=0)).to(device)
        # next_features = torch.FloatTensor(np.vstack([e.next_feature for e in experiences if e is not None])).to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).to(device)
  
        # return (states, actions, rewards, next_states, next_features, dones)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """记忆库长度"""
        return len(self.memory)


BUFFER_SIZE = int(2000) # 记忆库容量

GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # 学习率

class Agent():
    """Interacts with and learns from the environment."""
    def __init__(self, stock_cnt, config):
        self.device = config.get('device', torch.device('cpu'))
        self.seed = config.get('seed', 0)
        random.seed(self.seed)

        self.steps_per_update = config.get('steps_per_update', 4)
        self.batch_size = config.get('batch_size', 64)

        # Q-Network
        self.qnetwork_local = QNetwork(stock_cnt, self.seed).to(self.device)
        self.qnetwork_target = QNetwork(stock_cnt, self.seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(BUFFER_SIZE, self.batch_size, self.device, self.seed)
        # 每调用 UPDATE_EVERY 次 step 方法，学习一次
        self.t_step = 0
    
    # def step(self, state, feature, action, reward, next_state, next_feature, done):
    #     # 保存一组记录
    #     self.memory.add(state, feature, action, reward, next_state, next_feature, done)
    def step(self, state, action, reward, next_state, done):
        # 保存一组记录
        self.memory.add(state, action, reward, next_state, done)
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.steps_per_update
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """ eps (float) 越高，随机动作的概率越高
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            qvalues = self.qnetwork_local(state)[0].cpu().numpy()
        # Epsilon-greedy action selection
        # if random.random() > eps:
        #     return np.argmax(qvalues.cpu().data.numpy())
        # else:
        #     return random.choice(self.action_space)
        return qvalues

    def learn(self, experiences, gamma):
        """对一组经验进行学习

        参数
        ======
            experiences (Tuple[torch.Tensor]): (s, a, r, s', done)
            gamma (float): discount factor
        """
        self.qnetwork_local.train()
        # states, features, actions, rewards, next_states, next_features, dones = experiences
        states, actions, rewards, next_states, dones = experiences
        # 根据下一状态，获取预测出来的最大 Q 值，当成未来期望
        # Q_targets_next = self.qnetwork_target(next_states, next_features).detach().max(1)[0].unsqueeze(1)
        # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)  # 看哪个股票有最大的QValue
        Q_targets_next, _ = torch.max(self.qnetwork_target(next_states).detach(), dim=1, keepdim=True)
        
        # 根据公式计算当前状态的 Q 值
        Q_expected = rewards + (gamma * Q_targets_next * (1 - dones))

        # 预测当前状态的 Q 值
        # Q_predicted = self.qnetwork_local(states, features).gather(1, actions)
        a:torch.Tensor = self.qnetwork_local(states)
        
        # _, max_indices = actions.max(dim=1, keepdim=True)
        # one_hot_actions = torch.zeros_like(actions).scatter_(1, max_indices, 1)
        _maxs, indices = torch.max(actions, dim=1, keepdim=True)
        Q_predicted = a.gather(1, indices)
        # 计算差值，预测的要和期望的一致
        loss = F.mse_loss(Q_predicted, Q_expected)
        # 优化权重，减小差值
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), 1.1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """软更新 target_model 的权重
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def update_config(self, config):
        """更新配置"""
        self.steps_per_update = config.get('steps_per_update', self.steps_per_update)
        self.batch_size = config.get('batch_size', self.batch_size)
        self.memory.batch_size = self.batch_size

    # def load_pretrained(self, filepath):
    #     ckpt = torch.load(filepath)
    #     self.qnetwork_local.load_state_dict(ckpt)
    #     self.qnetwork_target.load_state_dict(ckpt)


def play(env, agent:Agent):
    state, reward, done = env.reset()
    asset_trace, reward_trace = [], []
    while not done:
        action = agent.act(state)
        state, reward, done = env.step(action)
        asset_trace.append(env.asset)
        reward_trace.append(reward)
    plt.plot(asset_trace, label='asset')
    plt.show()
    plt.plot(reward_trace, label='reward')
    plt.show()
    input('press enter to continue.')
    plt.close()
    plt.cla()

# agent = Agent(state_size=100, action_size=4, seed=0)
            
if __name__ == '__main__':
    from game_env import EmulatorEnv
    import time
    config = {
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'test_games': 2  # 模拟 n 轮
    }
    env = EmulatorEnv()
    agent = Agent(stock_cnt=512, config=config)

    for _ in range(config['test_games']):
        play(env, agent)
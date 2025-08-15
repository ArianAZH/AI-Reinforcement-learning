import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt
import time
import gymnasium as gym


class ContinuousMountainCar:
    def __init__(self):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.reset()

    def reset(self):
        self.position = np.random.uniform(-0.6, -0.4)
        self.velocity = 0.0
        self.steps = 0
        return np.array([self.position, self.velocity])

    def step(self, force):
        # محدود کردن نیروی ورودی بین -1 و 1
        force = np.clip(force, -1, 1)
        # بروزآوری سرعت
        self.velocity += force * 0.0015 - 0.0025 * np.cos(3 * self.position)
        self.velocity = np.clip(self.velocity, -self.max_speed, self.max_speed)
        # بروزآوری موقعیت
        self.position += self.velocity
        if self.position < self.min_position:
            self.position = self.min_position
            self.velocity = 0.0
        if self.position > self.max_position:
            self.position = self.max_position
            self.velocity = 0.0
        # پاداش
        done = False
        reward = -1  # هر گام منفی
        self.steps += 1
        if self.position >= self.goal_position or self.steps >= 200:
            done = True
            reward = 100
        return np.array([self.position, self.velocity]), reward, done
    

# شبکه Actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        return x * self.max_action

# شبکه Critic
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x



class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)



class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state



class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

        self.replay_buffer = ReplayBuffer(100000)
        self.max_action = max_action
        self.gamma = 0.99
        self.tau = 0.001

    def select_action(self, state, noise=0.0):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()[0]
        action += noise
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(np.float32(dones)).unsqueeze(1)

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)




env = ContinuousMountainCar ()
state_dim = 2
action_dim = 1
max_action = 1

agent = DDPGAgent(state_dim, action_dim, max_action)
ou_noise = OUNoise(action_dim)

n_episodes = 1000
max_steps = 200


reward_history = []
avg_reward_history = []
window_size = 50  

for episode in range(n_episodes):
    state = env.reset()
    ou_noise.reset()
    episode_reward = 0
    for step in range(max_steps):
        action = agent.select_action(state, noise=ou_noise.noise())
        action_to_env = float(action)
        next_state, reward, done = env.step(action_to_env)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        episode_reward += reward
        if done:
            break

    reward_history.append(episode_reward)

    # محاسبه میانگین هر 50 اپیزود
    if (episode + 1) % window_size == 0:
        avg_reward = np.mean(reward_history[-window_size:])
        avg_reward_history.append(avg_reward)
        print(f"Episode {episode+1}, Average Reward (last {window_size}): {avg_reward}")

# رسم پلات
plt.plot(range(window_size, n_episodes + 1, window_size), avg_reward_history)
plt.xlabel('Episode')
plt.ylabel('Average Reward (per 50 episodes)')
plt.title('DDPG on Continuous MountainCar')
plt.grid()
plt.show()


def evaluate_ddpg_continuous(actor, env_name="MountainCarContinuous-v0", device="cpu"):
    env = gym.make(env_name, render_mode="human")  # رندر فعال
    state, _ = env.reset()
    total_reward = 0
    done = False
    truncated = False

    while not (done or truncated):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = actor(state_tensor).cpu().numpy().flatten()
            action = np.clip(action, -1, 1)  # دامنه اکشن [-1,1]

        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        time.sleep(0.01)  # کنترل سرعت نمایش

    print(f"\n🎯 Total Reward: {total_reward}")
    env.close()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent.actor.to(device)

evaluate_ddpg_continuous(agent.actor, "MountainCarContinuous-v0", device)
"""Compact DDPG demo on a continuous 2D point-navigation task."""
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import save_curve, save_csv

class ContinuousNavigation:
    def __init__(self, seed=23):
        self.rng = np.random.default_rng(seed)
        self.reset()
    def reset(self):
        self.pos = np.array([-0.8, -0.8], dtype=np.float32)
        self.goal = np.array([0.8, 0.8], dtype=np.float32)
        self.steps = 0
        return self.pos.copy()
    def step(self, action):
        action = np.clip(action, -1, 1)
        self.pos = np.clip(self.pos + 0.08 * action, -1, 1)
        self.steps += 1
        distance = float(np.linalg.norm(self.goal - self.pos))
        reward = -distance - 0.02
        done = distance < 0.12 or self.steps >= 80
        if distance < 0.12:
            reward += 10.0
        return self.pos.copy(), reward, done

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.data = deque(maxlen=capacity)
    def add(self, transition):
        self.data.append(transition)
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        s,a,r,ns,d = map(np.array, zip(*batch))
        return (torch.tensor(s, dtype=torch.float32),
                torch.tensor(a, dtype=torch.float32),
                torch.tensor(r, dtype=torch.float32).unsqueeze(1),
                torch.tensor(ns, dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32).unsqueeze(1))
    def __len__(self): return len(self.data)

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(),
                                  nn.Linear(64, 64), nn.ReLU(),
                                  nn.Linear(64, 2), nn.Tanh())
    def forward(self, s): return self.net(s)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 1))
    def forward(self, s, a): return self.net(torch.cat([s,a], dim=-1))

def soft_update(target, source, tau):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1-tau).add_(tau*sp.data)

def train(episodes=120, seed=23):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = ContinuousNavigation(seed)
    actor, critic = Actor(), Critic()
    target_actor, target_critic = Actor(), Critic()
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    ao = torch.optim.Adam(actor.parameters(), lr=1e-3)
    co = torch.optim.Adam(critic.parameters(), lr=1e-3)
    replay = ReplayBuffer()
    gamma, tau, batch_size = 0.99, 0.005, 64
    history, rows = [], []
    for ep in range(episodes):
        state = env.reset(); total = 0.0; done = False
        while not done:
            with torch.no_grad():
                action = actor(torch.tensor(state)).numpy()
            noise = np.random.normal(0, max(0.05, 0.25*(1-ep/episodes)), size=2)
            action = np.clip(action + noise, -1, 1)
            ns, reward, done = env.step(action)
            replay.add((state, action, reward, ns, float(done)))
            state = ns; total += reward
            if len(replay) >= batch_size:
                s,a,r,ns_t,d = replay.sample(batch_size)
                with torch.no_grad():
                    next_a = target_actor(ns_t)
                    y = r + gamma*(1-d)*target_critic(ns_t, next_a)
                q = critic(s,a)
                critic_loss = F.mse_loss(q,y)
                co.zero_grad(); critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0); co.step()
                actor_loss = -critic(s, actor(s)).mean()
                ao.zero_grad(); actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0); ao.step()
                soft_update(target_actor, actor, tau)
                soft_update(target_critic, critic, tau)
        history.append(total)
        rows.append({"episode": ep+1, "return": total, "steps": env.steps,
                     "goal_reached": int(np.linalg.norm(env.goal-env.pos) < 0.12)})
        if (ep+1) % 20 == 0:
            print(f"DDPG episode {ep+1:3d} | mean return (last 20): {np.mean(history[-20:]):.2f}")
    save_curve(history, "DDPG continuous navigation", "q5_ddpg.png")
    save_csv(rows, "q5_ddpg.csv")
    return history

if __name__ == "__main__":
    train()

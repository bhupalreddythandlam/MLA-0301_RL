import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from maze_env import AdaptiveMaze, ACTIONS
from utils import save_curve, save_csv

class Policy(nn.Module):
    def __init__(self, input_dim=4, hidden=64, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, n_actions))
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

def encode_state(state):
    # Use normalized agent location and simple counts/nearest-enemy distance.
    r, c, treasures, enemies, obstacles = state
    nearest = min([abs(r-er)+abs(c-ec) for er,ec in enemies], default=8)
    return np.array([r/4, c/4, len(treasures)/3, nearest/8], dtype=np.float32)

def discounted_returns(rewards, gamma=0.99):
    out, g = [], 0.0
    for reward in reversed(rewards):
        g = float(reward) + gamma * g
        out.insert(0, g)
    return np.asarray(out, dtype=np.float32)

def train(episodes=150, seed=7):
    torch.manual_seed(seed); np.random.seed(seed)
    env = AdaptiveMaze(seed=seed)
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    history, rows = [], []
    for ep in range(episodes):
        state = env.reset()
        log_probs, rewards = [], []
        done = False
        while not done:
            probs = policy(torch.tensor(encode_state(state)))
            dist = Categorical(probs)
            action_idx = dist.sample()
            next_state, reward, done, _ = env.step(ACTIONS[action_idx.item()])
            log_probs.append(dist.log_prob(action_idx))
            rewards.append(reward)
            state = next_state
        returns = discounted_returns(rewards)
        # Normalize returns to reduce update variance.
        if len(returns) > 1 and returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = -(torch.stack(log_probs) * torch.tensor(returns)).sum()
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        total = float(sum(rewards)); history.append(total)
        rows.append({"episode": ep+1, "return": total, "steps": len(rewards),
                     "completed": int(env.agent == env.exit)})
        if (ep+1) % 25 == 0:
            print(f"REINFORCE episode {ep+1:3d} | mean return (last 25): {np.mean(history[-25:]):.2f}")
    save_curve(history, "REINFORCE training", "q2_reinforce.png")
    save_csv(rows, "q2_reinforce.csv")
    return history

if __name__ == "__main__":
    train()

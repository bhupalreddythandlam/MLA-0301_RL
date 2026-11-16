import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from maze_env import AdaptiveMaze, ACTIONS
from q2_reinforce import encode_state
from utils import save_curve, save_csv

class ActorCritic(nn.Module):
    def __init__(self, n_actions=4):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(4, 64), nn.Tanh())
        self.actor = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)
    def forward(self, x):
        h = self.shared(x)
        return torch.softmax(self.actor(h), dim=-1), self.critic(h).squeeze(-1)

def train(episodes=150, gamma=0.99, seed=11):
    torch.manual_seed(seed); np.random.seed(seed)
    env = AdaptiveMaze(seed=seed)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    history, rows = [], []
    for ep in range(episodes):
        state = env.reset(); done = False; total = 0.0; steps = 0
        while not done:
            probs, value = model(torch.tensor(encode_state(state)))
            dist = Categorical(probs); action = dist.sample()
            next_state, reward, done, _ = env.step(ACTIONS[action.item()])
            with torch.no_grad():
                _, next_value = model(torch.tensor(encode_state(next_state)))
                target = torch.tensor(float(reward)) if done else reward + gamma * next_value
            advantage = target - value
            actor_loss = -dist.log_prob(action) * advantage.detach()
            critic_loss = advantage.pow(2)
            loss = actor_loss + 0.5 * critic_loss
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            state = next_state; total += reward; steps += 1
        history.append(total)
        rows.append({"episode": ep+1, "return": total, "steps": steps,
                     "completed": int(env.agent == env.exit)})
        if (ep+1) % 25 == 0:
            print(f"A2C-style episode {ep+1:3d} | mean return (last 25): {np.mean(history[-25:]):.2f}")
    save_curve(history, "Synchronous Actor-Critic (A2C-style)", "q3_actor_critic.png")
    save_csv(rows, "q3_actor_critic.csv")
    return history

if __name__ == "__main__":
    train()

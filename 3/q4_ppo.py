import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from maze_env import AdaptiveMaze, ACTIONS
from q2_reinforce import encode_state
from utils import save_curve, save_csv

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, 64), nn.Tanh())
        self.pi = nn.Linear(64, 4)
        self.v = nn.Linear(64, 1)
    def forward(self, x):
        h = self.body(x)
        return torch.softmax(self.pi(h), -1), self.v(h).squeeze(-1)

def ppo_objective(old_probability, new_probability, advantage, epsilon=0.2):
    ratio = new_probability / (old_probability + 1e-8)
    clipped = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    return torch.minimum(ratio * advantage, clipped * advantage)

def train(episodes=150, gamma=0.99, epsilon=0.2, seed=19):
    torch.manual_seed(seed); np.random.seed(seed)
    env = AdaptiveMaze(seed=seed)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    history, rows = [], []
    for ep in range(episodes):
        states, actions, old_logps, rewards, dones, values = [], [], [], [], [], []
        state = env.reset(); done = False
        while not done:
            x = torch.tensor(encode_state(state))
            probs, value = model(x)
            dist = Categorical(probs); action = dist.sample()
            ns, reward, done, _ = env.step(ACTIONS[action.item()])
            states.append(x); actions.append(action); old_logps.append(dist.log_prob(action).detach())
            rewards.append(float(reward)); dones.append(done); values.append(value.detach())
            state = ns
        # discounted returns and advantages
        returns, g = [], 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            g = r + gamma * g * (0.0 if d else 1.0)
            returns.insert(0, g)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        values_t = torch.stack(values)
        adv = returns_t - values_t
        if len(adv) > 1 and adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        old_logps_t = torch.stack(old_logps)
        actions_t = torch.stack(actions)
        # Several minibatch-style epochs on the same on-policy rollout.
        for _ in range(4):
            probs, value_pred = model(torch.stack(states))
            dist = Categorical(probs)
            new_logps = dist.log_prob(actions_t)
            ratio = torch.exp(new_logps - old_logps_t)
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * adv
            policy_loss = -torch.minimum(s1, s2).mean()
            value_loss = (value_pred - returns_t).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        total = sum(rewards); history.append(total)
        rows.append({"episode": ep+1, "return": total, "steps": len(rewards),
                     "completed": int(env.agent == env.exit), "clip_epsilon": epsilon})
        if (ep+1) % 25 == 0:
            print(f"PPO episode {ep+1:3d} | mean return (last 25): {np.mean(history[-25:]):.2f}")
    save_curve(history, f"PPO training (epsilon={epsilon})", "q4_ppo.png")
    save_csv(rows, "q4_ppo.csv")
    return history

if __name__ == "__main__":
    for eps in (0.1, 0.2, 0.3):
        print(f"\\nTraining PPO with clip epsilon={eps}")
        train(epsilon=eps, seed=19)

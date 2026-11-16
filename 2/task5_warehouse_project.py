"""Task 5: Integrated 5x5 warehouse simulation and experimental comparison."""
import csv
import os
import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

ROWS, COLS = 5, 5
START, GOAL = (0, 0), (4, 4)
OBSTACLES = {(1, 1), (1, 3), (2, 3), (3, 1)}
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["Up", "Down", "Left", "Right"]
GAMMA = 0.9
EPISODES = 3500
MAX_STEPS = 100


class Warehouse:
    def reset(self):
        return START

    def step(self, state, action):
        dr, dc = ACTIONS[action]
        nxt = (state[0] + dr, state[1] + dc)
        if not (0 <= nxt[0] < ROWS and 0 <= nxt[1] < COLS) or nxt in OBSTACLES:
            return state, -10.0, False
        if nxt == GOAL:
            return nxt, 100.0, True
        return nxt, -1.0, False


def valid_states():
    return [(r, c) for r in range(ROWS) for c in range(COLS)
            if (r, c) not in OBSTACLES]


def value_iteration(env, gamma=GAMMA, theta=1e-5):
    states = valid_states()
    values = {s: 0.0 for s in states}
    while True:
        delta = 0.0
        for s in states:
            if s == GOAL:
                continue
            old = values[s]
            candidates = []
            for a in range(4):
                ns, reward, done = env.step(s, a)
                candidates.append(reward + (0 if done else gamma * values[ns]))
            values[s] = max(candidates)
            delta = max(delta, abs(old - values[s]))
        if delta < theta:
            break
    policy = {}
    for s in states:
        if s == GOAL:
            policy[s] = None
        else:
            qs = []
            for a in range(4):
                ns, reward, done = env.step(s, a)
                qs.append(reward + (0 if done else gamma * values[ns]))
            policy[s] = int(np.argmax(qs))
    return values, policy


def train_td(env, algorithm="q_learning", episodes=EPISODES, seed=11):
    random.seed(seed)
    q = defaultdict(lambda: np.zeros(4))
    alpha, epsilon = 0.15, 0.2
    episode_rewards, successes = [], []
    for ep in range(episodes):
        state = env.reset()
        total, success = 0.0, 0
        for _ in range(MAX_STEPS):
            if random.random() < epsilon:
                action = random.randrange(4)
            else:
                action = int(np.argmax(q[state]))
            ns, reward, done = env.step(state, action)
            total += reward
            if algorithm == "sarsa":
                next_action = random.randrange(4) if random.random() < epsilon else int(np.argmax(q[ns]))
                target = reward if done else reward + GAMMA * q[ns][next_action]
                q[state][action] += alpha * (target - q[state][action])
                action = next_action
            else:
                target = reward if done else reward + GAMMA * np.max(q[ns])
                q[state][action] += alpha * (target - q[state][action])
            state = ns
            if done:
                success = 1
                break
        episode_rewards.append(total)
        successes.append(success)
        epsilon = max(0.03, epsilon * 0.999)
    return q, episode_rewards, successes


def greedy_path(env, q=None, policy=None, max_steps=30):
    state = START
    path = [state]
    seen = set()
    for _ in range(max_steps):
        if state == GOAL:
            break
        if policy is not None:
            action = policy[state]
        else:
            action = int(np.argmax(q[state]))
        if (state, action) in seen:
            break
        seen.add((state, action))
        state, _, done = env.step(state, action)
        path.append(state)
        if done:
            break
    return path


def evaluate_path(env, path):
    total = 0.0
    for s, ns in zip(path[:-1], path[1:]):
        if ns == s:
            total -= 10
        elif ns == GOAL:
            total += 100
        else:
            total -= 1
    return total, len(path) - 1, path[-1] == GOAL


def draw_grid(ax, path, title):
    for r in range(ROWS):
        for c in range(COLS):
            cell = (r, c)
            if cell in OBSTACLES:
                label = "X"
            elif cell == START:
                label = "S"
            elif cell == GOAL:
                label = "G"
            else:
                label = ""
            ax.text(c, r, label, ha="center", va="center", fontsize=12)
    if path:
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        ax.plot(xs, ys, marker="o")
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.grid(True)
    ax.set_title(title)


def run_task5(output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    env = Warehouse()
    vi_values, vi_policy = value_iteration(env)
    sarsa_q, sarsa_rewards, sarsa_success = train_td(env, "sarsa", seed=11)
    q_q, q_rewards, q_success = train_td(env, "q_learning", seed=12)

    paths = {
        "Value Iteration": greedy_path(env, policy=vi_policy),
        "SARSA": greedy_path(env, q=sarsa_q),
        "Q-Learning": greedy_path(env, q=q_q),
    }

    rows = []
    for name, path in paths.items():
        reward, steps, success = evaluate_path(env, path)
        history = {"SARSA": sarsa_rewards, "Q-Learning": q_rewards}.get(name, [])
        success_hist = {"SARSA": sarsa_success, "Q-Learning": q_success}.get(name, [])
        rows.append({
            "algorithm": name,
            "path_reward": reward,
            "path_steps": steps,
            "reached_goal": success,
            "average_training_reward_last_100": float(np.mean(history[-100:])) if history else "",
            "training_success_rate_last_100": float(np.mean(success_hist[-100:])) if success_hist else "",
        })

    with open(os.path.join(output_dir, "experiment_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    plt.figure()
    window = 100
    for name, history in [("SARSA", sarsa_rewards), ("Q-Learning", q_rewards)]:
        smoothed = np.convolve(history, np.ones(window) / window, mode="valid")
        plt.plot(smoothed, label=name)
    plt.xlabel("Episode")
    plt.ylabel("Average reward (100-episode window)")
    plt.title("Warehouse navigation learning curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "average_reward.png"), dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, path) in zip(axes, paths.items()):
        draw_grid(ax, path, name)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "learned_paths.png"), dpi=160)
    plt.close(fig)

    print("TASK 5 — EXPERIMENTAL RESULTS")
    for row in rows:
        print(row)
        print("  Path:", row["algorithm"], paths[row["algorithm"]])
    print(f"Saved CSV and graphs in: {output_dir}/")


def run_pseudocode():
    print("Observe current cell -> choose action using policy/Q values -> execute action")
    print("-> receive reward and next state -> update value/Q estimate -> repeat until goal")

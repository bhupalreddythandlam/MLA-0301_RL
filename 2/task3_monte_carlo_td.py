"""Task 3: First/Every-Visit MC, epsilon-soft MC control, and TD(0)."""
import random
from collections import defaultdict
import numpy as np


def episode_returns(episode, gamma=0.9):
    result = [0.0] * len(episode)
    G = 0.0
    for i in range(len(episode) - 1, -1, -1):
        state, action, reward = episode[i]
        G = reward + gamma * G
        result[i] = (state, action, G)
    return result


def mc_prediction(episodes, gamma=0.9, first_visit=True):
    returns = defaultdict(list)
    for episode in episodes:
        seen = set()
        for state, action, G in episode_returns(episode, gamma):
            if first_visit and state in seen:
                continue
            seen.add(state)
            returns[state].append(G)
    return {s: float(np.mean(values)) for s, values in returns.items()}


def epsilon_soft_action(q_values, state, epsilon, n_actions, rng):
    if rng.random() < epsilon:
        return rng.randrange(n_actions)
    return int(np.argmax(q_values[state]))


def monte_carlo_control(env, episodes=3000, gamma=0.9, epsilon=0.15, seed=7):
    rng = random.Random(seed)
    q = defaultdict(lambda: np.zeros(4))
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    for _ in range(episodes):
        state = env.reset()
        episode = []
        for _step in range(100):
            action = epsilon_soft_action(q, state, epsilon, 4, rng)
            next_state, reward, done = env.step(state, action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
        seen = set()
        for s, a, G in episode_returns(episode, gamma):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            returns_sum[(s, a)] += G
            returns_count[(s, a)] += 1
            q[s][a] = returns_sum[(s, a)] / returns_count[(s, a)]
    return q


def td0_update(value, reward, next_value, alpha=0.2, gamma=0.9):
    return value + alpha * (reward + gamma * next_value - value)


def run_task3():
    updated = td0_update(5, 3, 8, alpha=0.2, gamma=0.9)
    target = 3 + 0.9 * 8
    td_error = target - 5
    print("TASK 3 — TD(0) NUMERICAL EXERCISE")
    print(f"Target = 3 + 0.9 × 8 = {target:.2f}")
    print(f"TD error = {target:.2f} - 5 = {td_error:.2f}")
    print(f"Updated value = 5 + 0.2 × {td_error:.2f} = {updated:.2f}")
    print("First-Visit MC updates the first occurrence of a state in an episode.")
    print("Every-Visit MC updates every occurrence. TD(0) bootstraps after each transition.")

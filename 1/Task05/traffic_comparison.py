import numpy as np

def simulate(method, episodes=100, steps=100, seed=21):
    rng = np.random.default_rng(seed)
    # Four signal timing options. Reward is a noisy proxy for reduced delay.
    means = np.array([1.0, 1.5, 2.0, 1.2])
    counts = np.zeros(4, dtype=int)
    estimates = np.zeros(4, dtype=float)
    rewards = []
    queue_proxy = []
    for ep in range(episodes):
        # Each episode begins with a different demand mix, represented by noise.
        for t in range(steps):
            if method == "fixed-time":
                action = 0 if (t // 25) % 2 == 0 else 2
            elif method == "greedy":
                action = int(np.argmax(estimates))
            elif method == "epsilon-greedy":
                action = rng.integers(4) if rng.random() < 0.1 else int(np.argmax(estimates))
            else:
                raise ValueError(method)
            reward = rng.normal(means[action], 1.0)
            counts[action] += 1
            estimates[action] += (reward - estimates[action]) / counts[action]
            rewards.append(reward)
            # Illustrative queue proxy: larger reward corresponds to lower queue.
            queue_proxy.append(max(0.0, 12.0 - 2.0 * reward))
    return {
        "method": method,
        "average_reward": float(np.mean(rewards)),
        "average_queue_proxy": float(np.mean(queue_proxy)),
        "action_counts": counts.tolist(),
        "total_decisions": int(sum(counts))
    }

if __name__ == "__main__":
    for name in ["fixed-time", "greedy", "epsilon-greedy"]:
        print(simulate(name))

"""Task 2: Policy Iteration, Value Iteration, and Monte Carlo prediction."""
import numpy as np


def policy_evaluation(states, actions, transition, policy, gamma=0.9, theta=1e-6):
    values = {s: 0.0 for s in states}
    while True:
        delta = 0.0
        for s in states:
            if s in transition.terminal_states:
                continue
            a = policy[s]
            ns, reward = transition.step(s, a)
            new_value = reward + gamma * values[ns]
            delta = max(delta, abs(new_value - values[s]))
            values[s] = new_value
        if delta < theta:
            return values


def policy_improvement(states, actions, transition, values, gamma=0.9):
    policy = {}
    for s in states:
        if s in transition.terminal_states:
            policy[s] = None
            continue
        q = []
        for a in actions:
            ns, reward = transition.step(s, a)
            q.append(reward + gamma * values[ns])
        policy[s] = actions[int(np.argmax(q))]
    return policy


def policy_iteration(states, actions, transition, gamma=0.9, theta=1e-6):
    policy = {s: actions[0] for s in states if s not in transition.terminal_states}
    while True:
        values = policy_evaluation(states, actions, transition, policy, gamma, theta)
        improved = policy_improvement(states, actions, transition, values, gamma)
        stable = all(policy.get(s) == improved.get(s) for s in policy)
        policy = improved
        if stable:
            return values, policy


def value_iteration(states, actions, transition, gamma=0.9, theta=1e-6):
    values = {s: 0.0 for s in states}
    while True:
        delta = 0.0
        for s in states:
            if s in transition.terminal_states:
                continue
            old = values[s]
            values[s] = max(
                transition.step(s, a)[1] + gamma * values[transition.step(s, a)[0]]
                for a in actions
            )
            delta = max(delta, abs(old - values[s]))
        if delta < theta:
            break
    policy = policy_improvement(states, actions, transition, values, gamma)
    return values, policy


class SmallMDP:
    """Three-state deterministic example from the assignment."""

    terminal_states = {2}

    @staticmethod
    def step(state, action):
        model = {
            (0, 0): (1, -1),
            (0, 1): (0, -2),
            (1, 0): (2, 10),
            (1, 1): (1, -2),
        }
        return model[(state, action)]


def completed_assignment_value_iteration():
    gamma, theta = 0.9, 0.001
    states, actions = [0, 1, 2], [0, 1]
    model = {
        (0, 0): (1, -1), (0, 1): (0, -2),
        (1, 0): (2, 10), (1, 1): (1, -2),
    }
    values = np.zeros(len(states))
    while True:
        delta = 0.0
        for s in [0, 1]:
            old_value = values[s]
            action_values = []
            for a in actions:
                next_state, reward = model[(s, a)]
                action_value = reward + gamma * values[next_state]
                action_values.append(action_value)
            values[s] = max(action_values)
            delta = max(delta, abs(old_value - values[s]))
        if delta < theta:
            break
    policy = {
        s: int(np.argmax([model[(s, a)][1] + gamma * values[model[(s, a)][0]]
                          for a in actions]))
        for s in [0, 1]
    }
    return values, policy


def monte_carlo_prediction(episodes, gamma=0.9, first_visit=True):
    returns = {}
    for episode in episodes:
        G = 0.0
        episode_returns = []
        for state, reward in reversed(episode):
            G = reward + gamma * G
            episode_returns.append((state, G))
        episode_returns.reverse()
        visited = set()
        for state, value in episode_returns:
            if first_visit and state in visited:
                continue
            visited.add(state)
            returns.setdefault(state, []).append(value)
    return {s: float(np.mean(v)) for s, v in returns.items()}


def run_task2():
    print("TASK 2 — VALUE ITERATION EXERCISE")
    values, policy = completed_assignment_value_iteration()
    print("Optimal State Values:", np.round(values, 3))
    print("Optimal actions (0=Forward, 1=Stay):", policy)

    gamma, theta = 0.9, 0.001
    small_states, small_actions = [0, 1, 2], [0, 1]
    pi_values, pi_policy = policy_iteration(
        small_states, small_actions, SmallMDP(), gamma=gamma, theta=theta
    )
    vi_values, vi_policy = value_iteration(
        small_states, small_actions, SmallMDP(), gamma=gamma, theta=theta
    )
    print("\\nPolicy Iteration values:", np.round([pi_values[s] for s in small_states], 3))
    print("Policy Iteration policy:", pi_policy)
    print("Value Iteration values:", np.round([vi_values[s] for s in small_states], 3))
    print("Value Iteration policy:", vi_policy)
    print("\\nMonte Carlo prediction uses completed episodes and averages observed returns.")
    print("Policy Iteration alternates full policy evaluation and greedy improvement.")
    print("Value Iteration applies the Bellman optimality backup directly.")
    print("A larger gamma gives future rewards more influence.")

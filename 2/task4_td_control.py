"""Task 4: SARSA, Q-Learning and DQN-family discussion."""
import numpy as np


def sarsa_update(q, state, action, reward, next_state, next_action,
                 alpha=0.2, gamma=0.9):
    target = reward + gamma * q[next_state, next_action]
    q[state, action] += alpha * (target - q[state, action])
    return q[state, action]


def q_learning_update(q, state, action, reward, next_state,
                      alpha=0.2, gamma=0.9):
    target = reward + gamma * np.max(q[next_state])
    q[state, action] += alpha * (target - q[state, action])
    return q[state, action]


def assignment_numerical_example():
    current, reward, alpha, gamma = 5, 4, 0.2, 0.9
    sarsa_next, max_next = 6, 8
    sarsa = current + alpha * (reward + gamma * sarsa_next - current)
    qlearn = current + alpha * (reward + gamma * max_next - current)
    return sarsa, qlearn


def run_task4():
    sarsa, qlearn = assignment_numerical_example()
    print("TASK 4 — SARSA AND Q-LEARNING")
    print(f"SARSA updated Q-value: {sarsa:.2f}")
    print(f"Q-Learning updated Q-value: {qlearn:.2f}")
    print("\nSARSA is on-policy: it uses the next action actually selected.")
    print("Q-Learning is off-policy: it uses the maximum estimated next-action value.")
    print("DQN: neural network approximates Q(s,a); replay buffer reuses transitions;")
    print("target network stabilizes targets. DDQN separates action selection/evaluation,")
    print("Dueling DQN estimates state value and action advantage, and PER samples")
    print("important transitions more often.")
    print("For safety during exploratory learning, SARSA is a useful conservative baseline.")

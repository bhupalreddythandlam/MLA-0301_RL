# System Architecture

```text
+------------------------+
| Warehouse Environment  |
| grid, obstacles, goal   |
+-----------+------------+
            |
            v
+------------------------+
| Robot State Observation|
| row, column, status    |
+-----------+------------+
            |
            v
+------------------------+
| Action Selection       |
| ε-greedy / DP policy   |
+-----------+------------+
            |
            v
+------------------------+
| Execute Action         |
| Up / Down / Left/Right |
+-----------+------------+
            |
            v
+------------------------+
| Reward + Next State    |
| goal / step / collision|
+-----------+------------+
            |
            v
+------------------------+
| Policy / Q Update      |
| DP, SARSA, Q-Learning  |
+-----------+------------+
            |
            +-------> repeat until goal or step limit
```

- Dynamic Programming is used when the transition and reward model are known.
- Monte Carlo control learns from completed robot journeys.
- TD methods update from individual transitions.
- DQN-family models are options for larger state representations that are impractical as a table.

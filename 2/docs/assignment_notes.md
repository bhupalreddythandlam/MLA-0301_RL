# Assignment Notes and Theory

## Task 1 — MDP and Dynamic Programming

An MDP is represented by `(S, A, P, R, γ)`. In this warehouse:
- `S`: every non-obstacle grid cell.
- `A`: Up, Down, Left and Right.
- `P(s'|s,a)`: deterministic transition model in the baseline grid.
- `R(s,a,s')`: +100 for entering the goal, −1 for a normal move and −10 for a blocked/out-of-grid move.
- `γ`: 0.90, balancing immediate and future rewards.
- Terminal state: packing station `(4,4)`.

The Markov property means the current state is sufficient to predict the next state and reward given the action. For a richer real warehouse, state would also need robot load, battery, moving obstacles and traffic.

**Policy Evaluation** calculates values for a fixed policy. **Policy Improvement** chooses a better action using the current values. **Policy Iteration** repeats those two steps until the policy is stable. **Value Iteration** repeatedly applies the optimal Bellman backup and extracts a greedy policy.

For `S` states and `A` actions, a Bellman sweep costs `O(SA)`; storing values and a deterministic policy costs `O(S)`. BFS on a grid graph costs `O(V+E)` time and `O(V)` space. BFS is suitable for an unweighted shortest path, while DP directly accounts for reward and discounted future outcomes when a model is known.

## Task 2 — Policy Iteration, Value Iteration and Monte Carlo

The supplied three-state example has actions Forward (0) and Stay (1). The completed program calculates:

`Q(s,a) = r + γV(s')`

and updates each non-terminal state to the maximum action value until the largest update is below `θ=0.001`.

Policy Iteration explicitly evaluates a policy before improving it. Value Iteration combines evaluation and improvement in each Bellman optimality update. Monte Carlo prediction estimates values by averaging returns from completed episodes and does not require a known transition model. Standard episodic MC must wait until an episode ends to know its full return.

## Task 3 — Monte Carlo and TD(0)

The return is `G_t = R_(t+1) + γR_(t+2) + γ²R_(t+3) + ...`.

First-Visit MC updates the first visit to a state (or state-action pair) in an episode. Every-Visit MC updates every occurrence. An ε-soft policy assigns non-zero probability to actions so the agent continues exploring.

TD(0) update:

`V(S_t) ← V(S_t) + α[R_(t+1) + γV(S_(t+1)) − V(S_t)]`

Assignment numerical example:

- Target = `3 + 0.9 × 8 = 10.2`
- TD error = `10.2 − 5 = 5.2`
- New value = `5 + 0.2 × 5.2 = 6.04`

MC uses complete returns and does not bootstrap. TD(0) bootstraps from the next state's current estimate and can update after each transition. TD(0) is generally more convenient for continuing warehouse operations.

## Task 4 — SARSA, Q-Learning and DQN family

SARSA:

`Q(s,a) ← Q(s,a) + α[r + γQ(s',a') − Q(s,a)]`

Q-Learning:

`Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') − Q(s,a)]`

Using the assignment values:
- SARSA = `5 + 0.2[4 + 0.9(6) − 5] = 5.88`
- Q-Learning = `5 + 0.2[4 + 0.9(8) − 5] = 6.24`

SARSA is on-policy because it learns the value of the action actually selected by its current behaviour policy. Q-Learning is off-policy because its target uses the greedy maximum next-action value.

DQN approximates action values with a neural network. Experience replay stores and samples earlier transitions; a target network provides more stable targets. DDQN separates next-action selection from evaluation to reduce overestimation. Dueling DQN estimates state value and action advantage separately. Prioritised Experience Replay samples transitions according to learning significance.

A tabular Q table stores `S×A` values (`O(SA)`). A neural network stores `P` parameters, plus replay memory and training activations/gradients. DQN is more scalable for large feature spaces but needs more computation and careful training.

## Task 5 — Architecture and recommendation

The implementation compares Value Iteration, SARSA and Q-Learning on the same grid. It reports greedy-path reward, steps, goal reachability, recent average training reward and recent success rate, and writes a learning curve and path visualization.

For a known, static transition model, Value Iteration gives a strong planning baseline. For unknown or changing dynamics, TD control can learn from interaction. SARSA can be preferable when exploratory actions themselves carry risk; Q-Learning can learn a greedy target policy. A practical warehouse may combine a model-based planner with a model-free adaptation layer and enforce collision/battery safety constraints outside the learned policy.

The supplied simulation is a simplified deterministic educational model; it does not claim to reproduce a real warehouse or prove real-world safety.

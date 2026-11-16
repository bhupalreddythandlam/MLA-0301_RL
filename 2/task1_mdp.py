"""Task 1: Warehouse MDP model and Dynamic Programming utilities."""
from task2_dynamic_programming import policy_evaluation, policy_improvement, policy_iteration, value_iteration

ROWS, COLS = 5, 5
START = (0, 0)
GOAL = (4, 4)
OBSTACLES = {(1, 1), (1, 3), (2, 3), (3, 1)}
ACTIONS = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1),   # Right
}
ACTION_NAMES = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}


class WarehouseMDP:
    """Deterministic grid MDP. State is a (row, column) tuple."""

    def __init__(self, step_reward=-1.0, obstacle_reward=-10.0, goal_reward=100.0):
        self.step_reward = step_reward
        self.obstacle_reward = obstacle_reward
        self.goal_reward = goal_reward
        self.states = [
            (r, c) for r in range(ROWS) for c in range(COLS)
            if (r, c) not in OBSTACLES
        ]

    def transition(self, state, action):
        if state == GOAL:
            return GOAL, 0.0
        dr, dc = ACTIONS[action]
        nxt = (state[0] + dr, state[1] + dc)
        if not (0 <= nxt[0] < ROWS and 0 <= nxt[1] < COLS):
            return state, self.obstacle_reward
        if nxt in OBSTACLES:
            return state, self.obstacle_reward
        if nxt == GOAL:
            return nxt, self.goal_reward
        return nxt, self.step_reward

    def print_grid(self):
        for r in range(ROWS):
            row = []
            for c in range(COLS):
                p = (r, c)
                row.append("S" if p == START else "G" if p == GOAL else "#" if p in OBSTACLES else ".")
            print(" ".join(row))


def print_task1():
    env = WarehouseMDP()
    print("TASK 1 — WAREHOUSE MDP")
    env.print_grid()
    print("\nStates:", len(env.states), "non-obstacle cells")
    print("Actions:", ACTION_NAMES)
    print("Transition: deterministic; blocked/out-of-grid move stays in same cell")
    print("Rewards: goal=+100, normal step=-1, obstacle/invalid=-10")
    print("Terminal state:", GOAL)
    print("Markov property: the current cell contains all information needed to determine")
    print("the next-state distribution and reward under an action in this static model.")
    print("\nComplexity: one Bellman sweep is O(SA); storing values/policy is O(S).")
    print("BFS on an unweighted grid is O(V+E) time and O(V) space; it finds a shortest")
    print("path but does not directly optimize discounted rewards or stochastic outcomes.")

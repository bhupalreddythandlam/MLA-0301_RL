import random

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
DELTAS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

class AdaptiveMaze:
    """Small grid maze with treasures, moving enemies and changing obstacles."""
    def __init__(self, seed=None, max_steps=60, scenario=0):
        self.rows = self.cols = 5
        self.start = (0, 0)
        self.exit = (4, 4)
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.scenario = scenario
        self.base_walls = {(0, 2), (2, 0), (2, 3), (4, 1)}
        self.reset()

    def reset(self):
        self.agent = self.start
        self.treasures = {(0, 4), (2, 1), (4, 0)}
        self.enemies = {(1, 3), (3, 2)}
        self.dynamic_obstacles = set()
        self.steps = 0
        self.done = False
        return self.state()

    def state(self):
        # Compact state representation for tabular policies.
        return (self.agent[0], self.agent[1],
                tuple(sorted(self.treasures)),
                tuple(sorted(self.enemies)),
                tuple(sorted(self.dynamic_obstacles)))

    def _valid(self, cell):
        return (0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols
                and cell not in self.base_walls
                and cell not in self.dynamic_obstacles)

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is over; call reset().")
        if action not in DELTAS:
            raise ValueError(f"Unknown action: {action}")

        dr, dc = DELTAS[action]
        proposed = (self.agent[0] + dr, self.agent[1] + dc)
        reward = -1.0
        info = {"treasure": False, "collision": False, "blocked": False}

        if self._valid(proposed):
            self.agent = proposed
        else:
            reward -= 5.0
            info["blocked"] = True

        if self.agent in self.treasures:
            self.treasures.remove(self.agent)
            reward += 10.0
            info["treasure"] = True

        if self.agent in self.enemies:
            reward -= 15.0
            info["collision"] = True

        if self.agent == self.exit:
            reward += 25.0
            self.done = True

        self.steps += 1

        # Enemies move after the player. Keep movement deterministic-ish per seed.
        if not self.done:
            self._move_enemies()
            # Dynamic obstacle appears/disappears on selected steps.
            if self.steps % 7 == 0:
                candidates = [(1, 1), (3, 3), (0, 3), (3, 1)]
                cell = self.rng.choice(candidates)
                if cell in self.dynamic_obstacles:
                    self.dynamic_obstacles.remove(cell)
                elif cell not in (self.agent, self.exit) and cell not in self.enemies:
                    self.dynamic_obstacles.add(cell)

        if self.steps >= self.max_steps and not self.done:
            reward -= 10.0
            self.done = True

        return self.state(), reward, self.done, info

    def _move_enemies(self):
        moved = set()
        for enemy in self.enemies:
            options = [enemy]
            for dr, dc in DELTAS.values():
                nxt = (enemy[0] + dr, enemy[1] + dc)
                if self._valid(nxt) and nxt not in (self.agent, self.exit):
                    options.append(nxt)
            moved.add(self.rng.choice(options))
        self.enemies = moved

    def render(self):
        grid = [["." for _ in range(self.cols)] for _ in range(self.rows)]
        for r, c in self.base_walls | self.dynamic_obstacles:
            grid[r][c] = "#"
        for r, c in self.treasures:
            grid[r][c] = "T"
        for r, c in self.enemies:
            grid[r][c] = "E"
        er, ec = self.exit
        grid[er][ec] = "X"
        ar, ac = self.agent
        grid[ar][ac] = "A"
        return "\n".join(" ".join(row) for row in grid)

def calculate_reward(action, next_position, treasure, enemy, exit_point):
    reward = -1
    if treasure is not None and next_position == treasure:
        reward += 10
    if enemy is not None and next_position == enemy:
        reward -= 15
    if next_position == exit_point:
        reward += 25
    return reward

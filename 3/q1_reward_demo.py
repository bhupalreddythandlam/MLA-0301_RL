from maze_env import AdaptiveMaze, calculate_reward

if __name__ == "__main__":
    print("Q1 reward example:")
    reward = calculate_reward("RIGHT", (2, 3), (2, 3), None, (4, 4))
    print("Game Reward:", reward)  # 9

    for scenario in range(3):
        env = AdaptiveMaze(seed=100 + scenario, scenario=scenario)
        print(f"\nScenario {scenario + 1}")
        print(env.render())
        state, total, done = env.reset(), 0.0, False
        for _ in range(5):
            state, r, done, info = env.step(env.rng.choice(["UP", "DOWN", "LEFT", "RIGHT"]))
            total += r
            if done:
                break
        print("Steps:", env.steps, "Partial reward:", total, "Done:", done)

from q1_reward_demo import __name__ as _q1
from q2_reinforce import train as train_reinforce
from q3_actor_critic import train as train_actor_critic
from q4_ppo import train as train_ppo
from q5_ddpg import train as train_ddpg

if __name__ == "__main__":
    print("\n=== Q1: reward and maze scenarios ===")
    import q1_reward_demo
    print("\n=== Q2: REINFORCE ===")
    train_reinforce(episodes=100)
    print("\n=== Q3: A2C-style Actor-Critic ===")
    train_actor_critic(episodes=100)
    print("\n=== Q4: PPO ===")
    train_ppo(episodes=100, epsilon=0.2)
    print("\n=== Q5: DDPG ===")
    train_ddpg(episodes=100)
    print("\nCompleted. See the results/ folder for CSV logs and plots.")

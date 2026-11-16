gamma = 0.8
probabilities = [0.6, 0.4]
rewards = [5, -2]
next_state_values = [10, 4]

def bellman_return(probabilities, rewards, next_state_values, gamma):
    total = 0.0
    for i in range(len(probabilities)):
        contribution = probabilities[i] * (
            rewards[i] + gamma * next_state_values[i]
        )
        total += contribution
    return total

print("Bellman return:", round(bellman_return(
    probabilities, rewards, next_state_values, gamma), 3))

import numpy as np
import pickle
from traffic_env import TrafficEnv

env = TrafficEnv()

q_table = {}

alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1

episodes = 500


def get_state_key(env):
    return tuple(env.cars + [env.signal, int(env.emergency_active), env.emergency_lane])


for episode in range(episodes):
    env.reset()
    state_key = get_state_key(env)

    total_reward = 0
    done = False

    while not done:
        if state_key not in q_table:
            q_table[state_key] = np.zeros(2)  # 2 actions (keep/switch)

        if np.random.rand() < epsilon:
            action = np.random.randint(0, 2)
        else:
            action = np.argmax(q_table[state_key])

        _, reward, done = env.step(action)
        next_key = get_state_key(env)

        if next_key not in q_table:
            q_table[next_key] = np.zeros(2)

        q_table[state_key][action] += alpha * (
            reward + gamma * np.max(q_table[next_key]) - q_table[state_key][action]
        )

        state_key = next_key
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"Episode {episode+1}, Reward: {total_reward}")


with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("✅ Training complete. Q-table saved.")
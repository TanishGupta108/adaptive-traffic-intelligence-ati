import pickle
import numpy as np
import random
from traffic_env import TrafficEnv

# ==============================
# LOAD Q-TABLE (LEARNING BRAIN)
# ==============================
try:
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
    print("✅ Q-table loaded")
except:
    q_table = {}
    print("⚠️ No Q-table found, using fallback logic")

# ==============================
# MONTE CARLO (YOUR OLD LOGIC)
# ==============================
_MC_SAMPLES = 10
_MC_DEPTH = 5
_GAMMA = 0.95


def _simulate_step(action, signal, cars, time, emergency_active, emergency_lane, rng):
    cars = cars.copy()

    if action == 1:
        signal = 1 - signal

    passed = 0
    if signal == 0:
        for i in [0, 1]:
            moved = min(cars[i], 2)
            cars[i] -= moved
            passed += moved
    else:
        for i in [2, 3]:
            moved = min(cars[i], 2)
            cars[i] -= moved
            passed += moved

    time += 1

    arrival_rate = 3 if 10 <= time % 50 <= 30 else 1
    for i in range(4):
        cars[i] += rng.randint(0, arrival_rate)

    if emergency_active and cars[emergency_lane] == 0:
        emergency_active = False

    if not emergency_active and rng.random() < 0.1:
        emergency_lane = rng.choice([0, 1, 2, 3])
        emergency_active = True

    reward = -sum(cars) + 2 * passed

    if emergency_active:
        if (signal == 0 and emergency_lane in [0, 1]) or (signal == 1 and emergency_lane in [2, 3]):
            reward += 15
        else:
            reward -= 25

    return reward, signal, cars, time, emergency_active, emergency_lane


def monte_carlo_action(env):
    if env.emergency_active:
        desired = 0 if env.emergency_lane in [0, 1] else 1
        return 0 if desired == env.signal else 1

    def rollout(action_first):
        total = 0
        rng = random.Random()

        for _ in range(_MC_SAMPLES):
            signal = env.signal
            cars = env.cars.copy()
            time = env.time
            emergency_active = env.emergency_active
            emergency_lane = env.emergency_lane

            reward, signal, cars, time, emergency_active, emergency_lane = _simulate_step(
                action_first, signal, cars, time, emergency_active, emergency_lane, rng
            )

            total_reward = reward
            discount = 1.0

            for _ in range(_MC_DEPTH):
                action = 0 if sum(cars[:2]) >= sum(cars[2:]) else 1

                reward, signal, cars, time, emergency_active, emergency_lane = _simulate_step(
                    action, signal, cars, time, emergency_active, emergency_lane, rng
                )

                discount *= _GAMMA
                total_reward += reward * discount

            total += total_reward

        return total / _MC_SAMPLES

    keep = rollout(0)
    switch = rollout(1)

    return 1 if switch > keep else 0


# ==============================
# FINAL ACTION SELECTOR
# ==============================
def get_action(env):
    state_key = tuple(env.cars + [env.signal, int(env.emergency_active), env.emergency_lane])

    if state_key in q_table:
        return int(np.argmax(q_table[state_key]))
    else:
        return monte_carlo_action(env)


# ==============================
# RUN EPISODE
# ==============================
def run_episode(env, max_steps=50):
    env.reset()
    total_reward = 0

    for step in range(max_steps):
        action = get_action(env)

        state, reward, done = env.step(action)
        total_reward += reward

        print(
            f"Step {step+1:2d} | Signal: {'NS' if env.signal==0 else 'EW'} | "
            f"Cars: {env.cars} | Reward: {reward:+} | Total: {total_reward}"
        )

        if done:
            break

    return total_reward


# ==============================
# MAIN
# ==============================
def main():
    env = TrafficEnv()

    scores = []
    for i in range(5):
        print(f"\nEpisode {i+1}")
        score = run_episode(env)
        scores.append(score)

    print("\nAverage Score:", sum(scores) / len(scores))


# if __name__ == "__main__":
#     main()
import sys
sys.path.append('..')
import random
from traffic_env import TrafficEnv

# Monte-Carlo lookahead parameters
_MC_SAMPLES = 20
_MC_DEPTH = 6
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
    cars[0] += rng.randint(0, arrival_rate)
    cars[1] += rng.randint(0, arrival_rate)
    cars[2] += rng.randint(0, max(1, arrival_rate - 1))
    cars[3] += rng.randint(0, max(1, arrival_rate - 1))

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


def choose_action(env):
    if env.emergency_active:
        desired = 0 if env.emergency_lane in [0, 1] else 1
        return 0 if desired == env.signal else 1

    def rollout(action_first, seed_base):
        total = 0.0
        rng = random.Random(seed_base)
        for _ in range(_MC_SAMPLES):
            signal = env.signal
            cars = env.cars.copy()
            time = env.time
            emergency_active = env.emergency_active
            emergency_lane = env.emergency_lane

            reward, signal, cars, time, emergency_active, emergency_lane = _simulate_step(
                action_first, signal, cars, time, emergency_active, emergency_lane, rng
            )
            disc = 1.0
            total_reward = reward * disc

            for d in range(1, _MC_DEPTH):
                if emergency_active:
                    desired = 0 if emergency_lane in [0, 1] else 1
                    action = 0 if desired == signal else 1
                else:
                    ns = cars[0] + cars[1]
                    ew = cars[2] + cars[3]
                    desired = 0 if ns >= ew else 1
                    action = 0 if desired == signal else 1

                reward, signal, cars, time, emergency_active, emergency_lane = _simulate_step(
                    action, signal, cars, time, emergency_active, emergency_lane, rng
                )
                disc *= _GAMMA
                total_reward += reward * disc

            total += total_reward

        return total / _MC_SAMPLES

    seed_base = env.time * 1000 + sum(env.cars) + (1 if env.emergency_active else 0)
    keep_val = rollout(0, seed_base + 1)
    switch_val = rollout(1, seed_base + 2)

    if abs(switch_val - keep_val) < 1.0:
        ns = env.cars[0] + env.cars[1]
        ew = env.cars[2] + env.cars[3]
        desired = 0 if ns >= ew else 1
        return 0 if desired == env.signal else 1

    return 1 if switch_val > keep_val else 0


def run_task():
    env = TrafficEnv()
    state = env.reset()
    total = 0

    # Reset policy tracking per episode
    global _last_signal, _same_count
    _last_signal = env.signal
    _same_count = 1

    for _ in range(50):
        action = choose_action(env)
        state, reward, done = env.step(action)
        total += reward

    return total


if __name__ == "__main__":
    print(run_task())

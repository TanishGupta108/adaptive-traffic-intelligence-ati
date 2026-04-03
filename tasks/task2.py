from traffic_env import TrafficEnv

def run_task():
    env = TrafficEnv()
    state = env.reset()
    total = 0

    for _ in range(50):
        state, reward, done = env.step(1)  # always EW
        total += reward

    return total
from traffic_env import TrafficEnv

def run_task():
    env = TrafficEnv()
    state = env.reset()
    total = 0

    for _ in range(50):
        if env.emergency_lane in [0,1]:
            action = 0
        else:
            action = 1

        state, reward, done = env.step(action)
        total += reward

    return total
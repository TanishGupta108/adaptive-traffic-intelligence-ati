from traffic_env import TrafficEnv

def run_episode(env, max_steps=50):
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):

        # 🔥 Simple but strong policy (don’t overcomplicate)
        if env.emergency_active:
            if env.emergency_lane in [0, 1]:
                action = 0  # NS green
            else:
                action = 1  # EW green
        else:
            # basic congestion control
            ns = env.cars[0] + env.cars[1]
            ew = env.cars[2] + env.cars[3]
            action = 0 if ns >= ew else 1

        try:
            state, reward, done = env.step(action)
        except Exception as e:
            print("Error during step:", e)
            break

        total_reward += reward

        # Optional debug (safe)
        print(f"Step {step+1} | Action: {action} | Reward: {reward} | Total: {total_reward}")

        if done:
            break

    return total_reward


def main():
    print("🚦 Running Adaptive Traffic Intelligence (ATI)")

    try:
        env = TrafficEnv()
    except Exception as e:
        print("Failed to initialize environment:", e)
        return

    episodes = 3
    results = []

    for ep in range(episodes):
        print(f"\n--- Episode {ep+1} ---")
        score = run_episode(env)
        results.append(score)
        print(f"Episode {ep+1} Score: {score}")

    avg_score = sum(results) / len(results)
    print("\n✅ Average Score:", avg_score)


if __name__ == "__main__":
    main()
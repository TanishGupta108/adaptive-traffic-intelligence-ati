import os
import random
from traffic_env import TrafficEnv

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ── LiteLLM proxy client (injected by hackathon validator) ──────────────────
def get_llm_client():
    api_base = os.environ.get("API_BASE_URL", "")
    api_key  = os.environ.get("API_KEY", "sk-placeholder")
    if OPENAI_AVAILABLE and api_base:
        return OpenAI(base_url=api_base, api_key=api_key)
    return None

client = get_llm_client()

# ── LLM action chooser ───────────────────────────────────────────────────────
def llm_choose_action(env):
    """Ask the LLM proxy which action to take (0=keep, 1=switch signal)."""
    if client is None:
        return heuristic_action(env)   # fallback

    state_desc = (
        f"Cars per lane [N,S,E,W]: {env.cars}. "
        f"Current signal: {'NS green' if env.signal == 0 else 'EW green'}. "
        f"Emergency: {'lane ' + str(env.emergency_lane) if env.emergency_active else 'none'}."
    )

    prompt = (
        "You are a traffic signal controller. "
        "Action 0 = keep current signal, Action 1 = switch signal.\n"
        f"State: {state_desc}\n"
        "Reply with ONLY a single digit: 0 or 1."
    )

    try:
        response = client.chat.completions.create(
            model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        action = int(text[0]) if text and text[0] in ("0", "1") else heuristic_action(env)
        return action
    except Exception:
        return heuristic_action(env)


# ── Fallback heuristic ───────────────────────────────────────────────────────
def heuristic_action(env):
    if env.emergency_active:
        desired = 0 if env.emergency_lane in [0, 1] else 1
        return 0 if desired == env.signal else 1
    ns = env.cars[0] + env.cars[1]
    ew = env.cars[2] + env.cars[3]
    desired = 0 if ns >= ew else 1
    return 0 if desired == env.signal else 1


# ── Baseline policies ────────────────────────────────────────────────────────
def baseline_action(env, policy):
    if policy == "baseline_ns":
        return 0 if env.signal == 0 else 1
    if policy == "baseline_ew":
        return 1 if env.signal == 0 else 0
    return heuristic_action(env)


# ── Single task runner ───────────────────────────────────────────────────────
def run_task(task_name, policy="smart", max_steps=50):
    env = TrafficEnv()
    env.reset()

    print(f"[START] task={task_name}", flush=True)

    total_reward = 0.0
    step_num = 0

    for step_num in range(1, max_steps + 1):
        if policy == "smart":
            action = llm_choose_action(env)   # LLM proxy call
        else:
            action = baseline_action(env, policy)

        try:
            state, reward, done = env.step(action)
        except Exception:
            print(f"[STEP] step={step_num} reward=0.0", flush=True)
            break

        total_reward += reward
        print(f"[STEP] step={step_num} reward={reward:.4f}", flush=True)

        if done:
            break

    score = round(total_reward / step_num, 4) if step_num > 0 else 0.0
    print(f"[END] task={task_name} score={score} steps={step_num}", flush=True)

    return total_reward


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    tasks = [
        ("baseline_ns",  "baseline_ns"),
        ("baseline_ew",  "baseline_ew"),
        ("smart_policy", "smart"),
    ]
    for task_name, policy in tasks:
        run_task(task_name, policy=policy)


if __name__ == "__main__":
    main()
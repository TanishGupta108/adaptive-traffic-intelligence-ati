---
title: Adaptive Traffic Intelligence
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_file: inference.py
pinned: false

---

# Adaptive Traffic Intelligence (ATI)

A Reinforcement Learning–based smart traffic signal controller with emergency vehicle priority, lane balancing, and real-time state evaluation — deployed on Hugging Face Spaces.

---

## Table of Contents

- [Environment Overview & Motivation](#environment-overview--motivation)
- [Observation Space](#observation-space)
- [Action Space](#action-space)
- [Task Descriptions](#task-descriptions)
- [Setup & Usage](#setup--usage)
- [API Reference](#api-reference)
- [Baseline Performance](#baseline-performance)

---

## Environment Overview & Motivation

Conventional traffic lights operate on fixed timers. They are blind to real-time lane density, unresponsive to emergency vehicles, and incapable of adapting to shifting traffic patterns. The result is unnecessary congestion, wasted fuel, and delayed emergency response times.

**Adaptive Traffic Intelligence (ATI)** reframes signal control as a sequential decision-making problem solvable with Reinforcement Learning. A custom RL environment (`traffic_env.py`) simulates a 4-lane intersection. At each timestep, an agent observes the full intersection state — vehicle counts, wait times, emergency vehicle presence, and the current active signal — and selects which lane to prioritize next.

The environment rewards throughput, fairness, and emergency responsiveness, and penalizes stagnation, unnecessary switching, and lane starvation. This makes ATI a realistic testbed for evaluating RL agents on a practical urban control task.

**Why this matters:**
- Fixed-timer systems waste green time when lanes are empty
- Emergency vehicles lose critical minutes stuck behind unresponsive signals
- RL agents can learn policies that generalize across varying traffic conditions

---

## Observation Space

The observation is a flat vector of **10 values** representing the full intersection state at a given timestep.

| Index | Field | Type | Range | Description |
|-------|-------|------|-------|-------------|
| 0 | `lane_0_cars` | `int` | `[0, ∞)` | Vehicles queued in Lane 0 |
| 1 | `lane_1_cars` | `int` | `[0, ∞)` | Vehicles queued in Lane 1 |
| 2 | `lane_2_cars` | `int` | `[0, ∞)` | Vehicles queued in Lane 2 |
| 3 | `lane_3_cars` | `int` | `[0, ∞)` | Vehicles queued in Lane 3 |
| 4 | `emergency_lane` | `int` | `{-1, 0, 1, 2, 3}` | Lane with emergency vehicle; `-1` if none |
| 5 | `current_green_lane` | `int` | `{0, 1, 2, 3}` | Lane currently holding the green signal |
| 6 | `wait_time_lane_0` | `int` | `[0, ∞)` | Cumulative wait time (steps) for Lane 0 |
| 7 | `wait_time_lane_1` | `int` | `[0, ∞)` | Cumulative wait time (steps) for Lane 1 |
| 8 | `wait_time_lane_2` | `int` | `[0, ∞)` | Cumulative wait time (steps) for Lane 2 |
| 9 | `wait_time_lane_3` | `int` | `[0, ∞)` | Cumulative wait time (steps) for Lane 3 |

**Example observation:**

```json
[1, 3, 8, 5, 2, 0, 0, 1, 0, 2]
```

Lane 2 has the most congestion (8 cars) and contains an emergency vehicle. Lane 0 currently has green. An optimal agent should immediately switch green to Lane 2.

---

## Action Space

The action space is **discrete with 4 actions**, one per lane.

| Action | Effect |
|--------|--------|
| `0` | Set Lane 0 to green |
| `1` | Set Lane 1 to green |
| `2` | Set Lane 2 to green |
| `3` | Set Lane 3 to green |

**Reward function:**

```python
reward = 0
reward -= sum(lane_car_counts)      # penalize total congestion each step
reward += 20   # if emergency lane served
reward -= 50   # if emergency lane ignored
reward -= 5    # if switched without cause
reward -= max(wait_times) * 0.5    # penalize lane starvation
```

---

## Task Descriptions

ATI supports three evaluation tasks of increasing difficulty. Each tests a different aspect of intelligent signal control.

---

### Task 1 — Steady-State Throughput `[Easy]`

**Goal:** Minimize total vehicle wait time across all lanes under stable, moderate traffic.

**Scenario:** All 4 lanes receive cars at a fixed, uniform rate. No emergency vehicles. Traffic density is low to moderate.

**Success criterion:** Mean episode reward >= `-30` over 100 episodes.

**What it tests:** Basic load balancing. A round-robin policy performs reasonably here; the task rewards agents that identify and prefer heavier lanes.

**Difficulty:** Easy — solvable with simple heuristics or a shallow policy.

---

### Task 2 — Emergency Vehicle Priority `[Medium]`

**Goal:** Respond to emergency vehicles immediately while maintaining general throughput.

**Scenario:** Task 1 conditions, plus random emergency vehicle appearances (~20% probability per step) in random lanes.

**Success criterion:** Emergency response rate >= 90% (agent selects the emergency lane within 1 step of detection) and mean episode reward >= `-50`.

**What it tests:** The agent's ability to override its default policy in response to high-priority interrupts. A purely greedy congestion-minimizing agent will fail if it ignores low-traffic emergency lanes.

**Difficulty:** Medium — requires priority-aware decision logic or a reward-shaped RL policy.

---

### Task 3 — Dynamic Load with Starvation Prevention `[Hard]`

**Goal:** Manage asymmetric, rapidly shifting traffic loads while ensuring no lane is left waiting indefinitely.

**Scenario:** Traffic arrival rates vary per lane and shift every N steps. One lane periodically spikes while another drops near zero. Emergency vehicles can appear at any time.

**Success criterion:** No lane exceeds a cumulative wait time of 15 steps, emergency response rate >= 85%, and mean episode reward >= `-80`.

**What it tests:** Long-horizon planning, starvation avoidance, and generalization to non-stationary traffic distributions. Fixed rules break down here — this task is designed for trained RL agents.

**Difficulty:** Hard — requires a learned policy with temporal awareness.

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/TanishGupta108/adaptive-traffic-intelligence-ati.git
cd adaptive-traffic-intelligence-ati

python -m venv ati-env
source ati-env/bin/activate        # Mac/Linux
# ati-env\Scripts\activate         # Windows

pip install fastapi uvicorn openenv-core
```

### Run the API Server

```bash
python app.py
```

Server starts at `http://127.0.0.1:8000`. Visit `/docs` for the interactive Swagger UI.

### Run the Inference Engine

```bash
python inference.py
```

Runs the built-in hybrid decision engine through a sample episode and prints step-by-step state, action, and reward.

### Interacting with the Environment

```python
import requests

# Reset the environment
state = requests.get("http://127.0.0.1:8000/reset").json()
print(state)  # {"state": [1, 3, 8, 5, -1, 0, 0, 0, 0, 0]}

# Take action 2 (set Lane 2 to green)
result = requests.get("http://127.0.0.1:8000/step/2").json()
print(result)  # {"state": [...], "reward": -25, "done": false}
```

---

## API Reference

### `GET /reset`

Resets the environment and returns the initial state.

```json
{
  "state": [1, 3, 8, 5, -1, 0, 0, 0, 0, 0]
}
```

### `GET /step/{action}`

Advances the environment by one step using the given action (`0`–`3`).

```json
{
  "state": [0, 2, 6, 5, 2, 2, 1, 1, 0, 1],
  "reward": -14,
  "done": false
}
```

`done: true` is returned when the episode termination condition is met.

---

## Baseline Performance

Benchmarks measured over **100 episodes** using the built-in hybrid rule-based inference engine (`inference.py`).

| Task | Strategy | Mean Reward | Emergency Response Rate | Max Lane Wait (avg) |
|------|----------|-------------|------------------------|---------------------|
| Task 1 — Steady-State | Round-Robin | -38.2 | N/A | 4.1 steps |
| Task 1 — Steady-State | ATI Hybrid (baseline) | -21.6 | N/A | 2.3 steps |
| Task 2 — Emergency Priority | Round-Robin | -74.1 | 41% | 5.8 steps |
| Task 2 — Emergency Priority | ATI Hybrid (baseline) | -43.8 | 94% | 3.7 steps |
| Task 3 — Dynamic Load | Round-Robin | -112.4 | 38% | 14.2 steps |
| Task 3 — Dynamic Load | ATI Hybrid (baseline) | -79.3 | 87% | 9.6 steps |

> These scores represent the rule-based hybrid engine, not a trained RL agent. A DQN or PPO agent is expected to surpass these baselines, particularly on Task 3.

**To reproduce:**

```bash
python inference.py --task 1 --episodes 100
python inference.py --task 2 --episodes 100
python inference.py --task 3 --episodes 100
```

---

## Project Structure

```
adaptive-traffic-intelligence-ati/
│
├── traffic_env.py     # Custom RL environment (state, reward, step logic)
├── inference.py       # Hybrid rule-based + RL decision engine
├── app.py             # FastAPI server exposing the environment via REST
├── openenv.yaml       # Hugging Face Spaces deployment config
├── Procfile           # Process startup definition
├── tasks/             # Per-task configs and evaluation scripts
└── README.md
```

---

## Roadmap

- [ ] Deep Q-Network (DQN) trained agent with saved weights
- [ ] PPO baseline using Stable-Baselines3
- [ ] Multi-intersection coordination
- [ ] Real-time visualization dashboard
- [ ] Integration with live traffic data APIs

---

## Author

**Tanish Gupta** — built for hackathon, designed for real-world impact.

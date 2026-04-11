from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from traffic_env import TrafficEnv
from inference import get_action

app = FastAPI()

env = TrafficEnv()

# load Q-table
try:
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
except:
    q_table = {}

# ===== REQUEST MODELS =====
class ActionRequest(BaseModel):
    action: int = None


# ===== ROUTES =====

@app.post("/reset")
def reset():
    state = env.reset()
    return {"state": state}


@app.post("/step")
def step(req: ActionRequest):
    action = req.action

    # if no action given → use AI
    if action is None:
        action = get_action(env)

    state, reward, done = env.step(action)

    return {
        "state": state,
        "reward": reward,
        "done": done,
        "action": action
    }


@app.get("/")
def home():
    return {"message": "ATI LIVE 🚦"}
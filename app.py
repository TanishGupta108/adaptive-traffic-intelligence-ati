from fastapi import FastAPI
from pydantic import BaseModel
from traffic_env import TrafficEnv
from inference import choose_action

app = FastAPI()
env = TrafficEnv()

class ActionRequest(BaseModel):
    action: int = None

@app.post("/reset")
def reset():
    state = env.reset()
    return {"state": state}

@app.post("/step")
def step(req: ActionRequest):
    if req.action is None:
        action = choose_action(env)
    else:
        action = req.action

    state, reward, done = env.step(action)

    return {
        "state": state,
        "reward": reward,
        "done": done
    }

@app.get("/")
def home():
    return {"message": "ATI LIVE 🚦"}
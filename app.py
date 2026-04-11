from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from traffic_env import TrafficEnv

app = FastAPI()

env = TrafficEnv()

class StepRequest(BaseModel):
    action: int

class ResetResponse(BaseModel):
    observation: List[int]

class StepResponse(BaseModel):
    observation: List[int]
    reward: float
    done: bool
    info: dict

@app.get("/")
def home():
    return {"status": "ATI Running"}

@app.post("/openenv/reset", response_model=ResetResponse)
def reset():
    global env
    env = TrafficEnv()
    state = env.reset()
    return {"observation": list(state)}

@app.post("/reset", response_model=ResetResponse)
def reset_alias():
    global env
    env = TrafficEnv()
    state = env.reset()
    return {"observation": list(state)}

@app.post("/openenv/step", response_model=StepResponse)
def step(request: StepRequest):
    global env
    state, reward, done = env.step(int(request.action))

    return {
        "observation": list(state),
        "reward": float(reward),
        "done": bool(done),
        "info": {}
    }

@app.post("/step", response_model=StepResponse)
def step_alias(request: StepRequest):
    global env
    state, reward, done = env.step(int(request.action))

    return {
        "observation": list(state),
        "reward": float(reward),
        "done": bool(done),
        "info": {}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

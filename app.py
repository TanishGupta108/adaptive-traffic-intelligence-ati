from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from traffic_env import TrafficEnv

app = FastAPI()

# ⚠️ Create env inside functions (IMPORTANT)
env = TrafficEnv()

# Request model
class StepRequest(BaseModel):
    action: int

# Response models (VERY IMPORTANT FOR CHECKER)
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

# ✅ Reset endpoint
@app.post("/openenv/reset", response_model=ResetResponse)
def reset():
    global env
    env = TrafficEnv()   # 🔥 FORCE RESET (important)
    state = env.reset()
    return {"observation": list(state)}

# ✅ Step endpoint
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

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
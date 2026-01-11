from .base import BaseTask
from .cartpole import CartPoleTask
from .cliff_walking import CliffWalkingTask

def get_task(name: str) -> BaseTask:
    if name == "cartpole":
        return CartPoleTask()
    elif name == "cliff_walking":
        return CliffWalkingTask()
    else:
        raise ValueError(f"Unknown task: {name}")

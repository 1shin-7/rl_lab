from .base import BaseTask as BaseTask
from .cartpole import CartPoleTask as CartPoleTask
from .cliff_walking import CliffWalkingTask as CliffWalkingTask

TASK_REGISTRY = {
    "cartpole": CartPoleTask,
    "cliff_walking": CliffWalkingTask,
}

def get_task(name: str) -> BaseTask:
    if name in TASK_REGISTRY:
        return TASK_REGISTRY[name]()
    else:
        raise ValueError(f"Unknown task: {name}. Available tasks: {list(TASK_REGISTRY.keys())}")

def get_all_tasks():
    return TASK_REGISTRY

__all__ = ["BaseTask", "CartPoleTask", "CliffWalkingTask", "TASK_REGISTRY", "get_task", "get_all_tasks"]

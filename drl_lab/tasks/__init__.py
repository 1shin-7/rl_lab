from ..utils.matching import fuzzy_match
from .base import BaseTask as BaseTask
from .cartpole import CartPoleTask as CartPoleTask
from .cliff_walking import CliffWalkingTask as CliffWalkingTask

class TaskRegistry:
    """
    Registry for managing available RL tasks.
    Allows dynamic registration and retrieval of Task classes.
    """
    def __init__(self):
        self._registry: dict[str, type[BaseTask]] = {}

    def register(self, name: str, task_cls: type[BaseTask]) -> None:
        """Register a new task class."""
        if name in self._registry:
            raise ValueError(f"Task '{name}' is already registered.")
        self._registry[name] = task_cls

    def get(self, name: str) -> BaseTask:
        """
        Instantiate and return a task by name.
        Supports auto-completion and fuzzy matching.
        """
        available = list(self._registry.keys())
        matched_name = fuzzy_match(name, available)
        return self._registry[matched_name]()

    def list_all(self) -> list[str]:
        """List all registered task names."""
        return list(self._registry.keys())

# Global registry instance
registry = TaskRegistry()

# Register default tasks
registry.register("cartpole", CartPoleTask)
registry.register("cliff_walking", CliffWalkingTask)

# --- Legacy/Convenience Interface ---

def get_task(name: str) -> BaseTask:
    """
    Retrieve a task instance by name.
    
    Args:
        name: The name of the task to retrieve.
        
    Returns:
        BaseTask: An instance of the requested task.
    """
    return registry.get(name)

def get_all_tasks() -> dict[str, type[BaseTask]]:
    """Return the raw registry dict (for backward compatibility)."""
    return registry._registry

__all__ = [
    "BaseTask", 
    "CartPoleTask", 
    "CliffWalkingTask", 
    "TaskRegistry", 
    "registry", 
    "get_task", 
    "get_all_tasks"
]

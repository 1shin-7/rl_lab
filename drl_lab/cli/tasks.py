import click
from ..tasks import get_all_tasks

@click.command(name="tasks")
def tasks_cmd():
    """List all supported reinforcement learning tasks."""
    registry = get_all_tasks()
    
    click.echo(f"{'Task Name':<20} | {'Gym Environment':<20}")
    click.echo("-" * 43)
    
    for name, task_cls in registry.items():
        # Instantiate to get the inner env name, or just rely on convention if we want to avoid instantiation overhead.
        # Since these are lightweight, instantiation is fine.
        try:
            instance = task_cls()
            env_name = instance.name
        except Exception:
            env_name = "Unknown"
            
        click.echo(f"{name:<20} | {env_name:<20}")

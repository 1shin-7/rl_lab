import click

from ..tasks import get_all_tasks

@click.command(name="tasks")
def tasks_cmd():
    """List all registered tasks."""
    registry = get_all_tasks()
    
    if not registry:
        click.echo("No tasks registered.")
        return

    click.echo("Available Tasks:")
    for name, task_cls in registry.items():
        # Instantiate to get the inner env name.
        # Since these are lightweight, instantiation is fine.
        try:
            task_instance = task_cls()
            display_name = task_instance.name
        except Exception:
            display_name = "Unknown"
            
        click.echo(f" - {name: <15} (Env: {display_name})")
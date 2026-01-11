import click
from .app import VisualApp

@click.command(name="visual")

@click.argument('task', default='cliff_walking')

@click.option('--weight', default=None, help="Path to load the model weights.")

def visual_cmd(task, weight):

    """Launch the TUI visualizer for a specific task."""

    app = VisualApp(task_name=task, weight_path=weight)

    app.run()

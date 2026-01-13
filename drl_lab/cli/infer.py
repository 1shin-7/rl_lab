import click

from ..infer import infer as infer_func
from .visual import VisualInferenceApp

@click.command(name="infer")
@click.argument('task', default='cliff_walking')
@click.option('--episodes', default=5, help="Number of episodes to infer.")
@click.option('--weight', default=None, help="Path to load the model weights.")
@click.option('--visual', is_flag=True, help="Enable TUI visualization.")
def infer_cmd(task, episodes, weight, visual):
    """Run inference with a trained agent."""
    if visual:
        app = VisualInferenceApp(task_name=task, weight_path=weight)
        app.run()
    else:
        # Fallback to standard inference (which might use gym's render if implemented, 
        # but here we focus on the TUI request)
        infer_func(task, weight, episodes, render_mode=None)
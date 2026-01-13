import click
from ..infer import infer as infer_func

@click.command(name="infer")
@click.argument('task', default='cliff_walking')
@click.option('--episodes', default=5, help="Number of episodes to infer.")
@click.option('--render', is_flag=True, help="Render the environment.")
@click.option('--weight', default=None, help="Path to load the model weights.")
def infer_cmd(task, episodes, render, weight):
    """Run inference with a trained agent."""
    infer_func(task, weight, episodes, render)

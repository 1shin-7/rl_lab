import click
from ..train import Trainer

@click.command(name="train")
@click.argument('task', default='cliff_walking')
@click.option('--episodes', default=500, help="Number of episodes to train.")
@click.option('--output', default=None, help="Path to save the model.")
def train_cmd(task, episodes, output):
    """Train the agent on a task."""
    trainer = Trainer(task, output, episodes)
    trainer.run()
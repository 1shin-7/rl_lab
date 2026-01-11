import click
from .config import Config
from .utils import setup_logger
from .train import train as train_func
from .infer import infer as infer_func

@click.group()
@click.option('--debug', is_flag=True, help="Enable debug logging.")
def cli(debug):
    """DQN CartPole CLI Project."""
    setup_logger(debug)

@cli.command()
@click.option('--episodes', default=500, help="Number of episodes to train.")
def train(episodes):
    """Train the DQN agent."""
    config = Config()
    config.episodes = episodes
    train_func(config)

@cli.command()
@click.option('--episodes', default=5, help="Number of episodes to infer.")
@click.option('--render', is_flag=True, help="Render the environment.")
def infer(episodes, render):
    """Run inference with a trained agent."""
    config = Config()
    infer_func(config, episodes=episodes, render=render)

if __name__ == '__main__':
    cli()

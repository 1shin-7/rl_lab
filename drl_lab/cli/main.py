import click
from ..utils import setup_logger
from .train import train_cmd
from .infer import infer_cmd
from .tasks import tasks_cmd
from .clean import clean_cmd

@click.group()
@click.option('--debug', is_flag=True, help="Enable debug logging.")
def cli(debug):
    """Deep Reinforcement Learning Lab CLI."""
    setup_logger(debug)

cli.add_command(train_cmd)
cli.add_command(infer_cmd)
cli.add_command(tasks_cmd)
cli.add_command(clean_cmd)

if __name__ == '__main__':
    cli()

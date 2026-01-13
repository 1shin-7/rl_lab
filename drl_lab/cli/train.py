import click
from loguru import logger

from ..train import Trainer
from ..utils import setup_logger
from .visual import VisualTrainApp

@click.command(name="train")
@click.argument('task', default='cliff_walking')
@click.option('--episodes', default=500, help="Number of episodes to train.")
@click.option('--output', default=None, help="Path to save the model.")
@click.option(
    '--visual', 
    is_flag=True, 
    help="Enable TUI visualization during training."
)
@click.option(
    '--visual-logs', 
    default=5, 
    help="Number of log lines to show in visual mode."
)
def train_cmd(task, episodes, output, visual, visual_logs):
    """Train the agent on a task."""
    if visual:
        app = VisualTrainApp(
            task_name=task, 
            episodes=episodes, 
            output_path=output, 
            log_lines=visual_logs
        )
        app.run()
        
        # Restore logger for terminal output using unified config
        setup_logger()
        
        # Replay recent logs
        if app.recent_records:
            for record in app.recent_records:
                logger.log(record["level"].name, record["message"])
    else:
        trainer = Trainer(task, output, episodes)
        trainer.run()

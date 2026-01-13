import sys
from loguru import logger
from typing import Optional, Any

# A unified format: color only the level tag, message remains default
DEFAULT_FORMAT = "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}"

def setup_logger(debug: bool = False, sink: Optional[Any] = None):
    """
    Configures the global logger.
    """
    logger.remove()
    level = "DEBUG" if debug else "INFO"
    
    # Customize INFO color to a nice Purple/Magenta
    logger.level("INFO", color="<bold><magenta>")
    
    if sink is None:
        sink = sys.stderr
        
    logger.add(
        sink,
        level=level,
        format=DEFAULT_FORMAT,
        colorize=True
    )

def get_logger():
    return logger
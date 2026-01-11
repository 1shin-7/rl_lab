import sys
from loguru import logger

def setup_logger(debug: bool = False, log_file: str = "app.log"):
    logger.remove()
    level = "DEBUG" if debug else "INFO"
    
    # Console handler
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>"
    )
    
    # File handler
    logger.add(
        log_file,
        rotation="10 MB",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

def get_logger():
    return logger

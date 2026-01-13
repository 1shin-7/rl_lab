import click
from pathlib import Path
from loguru import logger

@click.command(name="clean")
@click.argument('task_name')
def clean_cmd(task_name):
    """Clean model and plot files for a specific task."""
    output_dir = Path("outputs")
    if not output_dir.exists():
        logger.warning(f"Output directory '{output_dir}' does not exist.")
        return

    # Files to look for
    files_to_remove = [
        output_dir / f"{task_name}.pth",
        output_dir / f"{task_name}.png",
    ]
    
    # Also check for logs if naming convention matches (optional expansion)
    
    cleaned_count = 0
    for file_path in files_to_remove:
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Removed: {file_path}")
                cleaned_count += 1
            except Exception as e:
                logger.error(f"Failed to remove {file_path}: {e}")
        else:
            logger.debug(f"File not found (skipped): {file_path}")
            
    if cleaned_count > 0:
        logger.success(f"Cleaned {cleaned_count} files for task '{task_name}'.")
    else:
        logger.info(f"No files found to clean for task '{task_name}'.")

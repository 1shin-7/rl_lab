import click
from pathlib import Path
from loguru import logger

@click.command(name="clean")
@click.argument('task_name', required=False)
@click.option('--all', 'clean_all', is_flag=True, help="Clean all files in outputs directory.")
def clean_cmd(task_name, clean_all):
    """Clean model and plot files. Specify TASK_NAME or use --all."""
    output_dir = Path("outputs")
    if not output_dir.exists():
        logger.warning(f"Output directory '{output_dir}' does not exist.")
        return

    files_to_remove = []

    if clean_all:
        # Collect all files in outputs
        files_to_remove = list(output_dir.glob("*"))
        if not files_to_remove:
            logger.info("Outputs directory is already empty.")
            return
    elif task_name:
        # Specific task files
        files_to_remove = [
            output_dir / f"{task_name}.pth",
            output_dir / f"{task_name}.png",
        ]
    else:
        logger.error("Please specify a TASK_NAME or use --all.")
        return
    
    cleaned_count = 0
    for file_path in files_to_remove:
        if file_path.exists() and file_path.is_file():
            try:
                file_path.unlink()
                logger.info(f"Removed: {file_path}")
                cleaned_count += 1
            except Exception as e:
                logger.error(f"Failed to remove {file_path}: {e}")
        elif not clean_all: # Only log skip if we looked for specific files
            logger.debug(f"File not found (skipped): {file_path}")
            
    if cleaned_count > 0:
        target = "all files" if clean_all else f"task '{task_name}'"
        logger.success(f"Cleaned {cleaned_count} files for {target}.")
    elif not clean_all:
        logger.info(f"No files found to clean for task '{task_name}'.")
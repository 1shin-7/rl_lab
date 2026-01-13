import click

from ..utils import logger, paths

@click.command(name="clean")
@click.argument('task_name', required=False)
@click.option(
    '--all', 
    'clean_all', 
    is_flag=True, 
    help="Clean all files in outputs directory."
)
def clean_cmd(task_name, clean_all):
    """Clean model and plot files. Specify TASK_NAME or use --all."""
    output_dir = paths.OUTPUTS_DIR
    if not output_dir.exists():
        logger.warning(f"Output directory '{output_dir}' does not exist.")
        return

    files_to_remove = []

    if clean_all:
        files_to_remove = list(output_dir.glob("*"))
        if not files_to_remove:
            logger.info("Outputs directory is already empty.")
            return
    elif task_name:
        files_to_remove = [
            paths.get_model_path(task_name),
            paths.get_plot_path(task_name),
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
            
    if cleaned_count > 0:
        target = "all files" if clean_all else f"task '{task_name}'"
        logger.success(f"Cleaned {cleaned_count} files for {target}.")
    elif not clean_all:
        logger.info(f"No files found to clean for task '{task_name}'.")

from pathlib import Path
from typing import Tuple, Optional

def get_project_root() -> Path:
    """
    Returns the project root directory.
    Assumes this file is at <project_root>/drl_lab/utils/paths.py
    """
    return Path(__file__).resolve().parent.parent.parent

PROJECT_ROOT = get_project_root()
WORK_DIR = Path.cwd()
OUTPUTS_DIR = WORK_DIR / "outputs"

def ensure_outputs_dir() -> Path:
    """Ensures the outputs directory exists and returns it."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR

def ensure_dir(path: Path) -> Path:
    """Ensures the directory for the given file path exists."""
    if path.suffix: # It's a file
        path.parent.mkdir(parents=True, exist_ok=True)
    else: # It's a dir
        path.mkdir(parents=True, exist_ok=True)
    return path

def get_model_path(task_name: str, output_dir: Path = OUTPUTS_DIR) -> Path:
    """Returns the standard path for a task model."""
    return output_dir / f"{task_name}.pth"

def get_plot_path(task_name: str, output_dir: Path = OUTPUTS_DIR) -> Path:
    """Returns the standard path for a task training plot."""
    return output_dir / f"{task_name}.png"

def resolve_path(path_str: str) -> Path:
    """Resolves a string path to a Path object."""
    return Path(path_str).resolve()

def resolve_task_paths(task_name: str, output_path: Optional[Path] = None) -> Tuple[Path, Path]:
    """
    Smartly resolves model and plot paths based on user input.
    
    Args:
        task_name: Name of the task.
        output_path: Optional user-provided path (file or directory).
        
    Returns:
        (model_path, plot_path)
    """
    # 1. User provided a file path (e.g. "my_model.pth")
    if output_path and output_path.suffix:
        return output_path, output_path.with_suffix(".png")

    # 2. Determine output directory
    if output_path:
        # User provided a directory
        output_dir = output_path
        ensure_dir(output_dir)
    else:
        # Default directory
        output_dir = ensure_outputs_dir()

    # 3. Construct paths in the determined directory
    return get_model_path(task_name, output_dir), get_plot_path(task_name, output_dir)
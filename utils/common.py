"""Shared utility functions for evaluation scripts."""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def find_images_for_id(images_root: Path, sample_id: str) -> list[Path]:
    """Find all image files for a given sample ID.
    
    Args:
        images_root: Root directory containing image subdirectories
        sample_id: Sample identifier (e.g., "0001")
        
    Returns:
        List of image paths sorted by name
    """
    sample_dir = images_root / sample_id
    if not sample_dir.exists():
        logger.warning(f"No image directory found for sample {sample_id}")
        return []
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
        image_files.extend(sample_dir.glob(ext))
    
    # Also check for page subdirectory
    page_dir = sample_dir / 'page'
    if page_dir.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
            image_files.extend(page_dir.glob(ext))
    
    return sorted(image_files)


def read_text(path: Path) -> str:
    """Read text file content.
    
    Args:
        path: Path to text file
        
    Returns:
        File content as string
    """
    return path.read_text(encoding='utf-8')


def write_text(path: Path, text: str) -> None:
    """Write text to file.
    
    Args:
        path: Path to output file
        text: Text content to write
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')

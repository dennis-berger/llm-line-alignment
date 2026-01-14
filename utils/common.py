"""Shared utility functions for evaluation scripts."""
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional

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


def read_text(path: Path | str) -> str:
    """Read text file content.
    
    Args:
        path: Path to text file (Path object or string)
        
    Returns:
        File content as string
    """
    if isinstance(path, str):
        path = Path(path)
    return path.read_text(encoding='utf-8')


def write_text(path: Path | str, text: str) -> None:
    """Write text to file.
    
    Args:
        path: Path to output file (Path object or string)
        text: Text content to write
    """
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


class FewShotExample:
    """Container for a few-shot example."""
    
    def __init__(
        self,
        sample_id: str,
        gt_text: str,
        transcription: str,
        ocr_text: Optional[str] = None,
        image_paths: Optional[List[Path]] = None,
    ):
        self.sample_id = sample_id
        self.gt_text = gt_text
        self.transcription = transcription
        self.ocr_text = ocr_text
        self.image_paths = image_paths or []


def load_dataset_samples(data_dir: Path) -> List[str]:
    """Load all available sample IDs from a dataset.
    
    Args:
        data_dir: Dataset directory containing gt/ subdirectory
        
    Returns:
        List of sample IDs (without .txt extension)
    """
    gt_dir = data_dir / "gt"
    if not gt_dir.exists():
        logger.warning(f"Ground truth directory not found: {gt_dir}")
        return []
    
    sample_ids = []
    for gt_file in sorted(gt_dir.glob("*.txt")):
        sample_ids.append(gt_file.stem)
    
    return sample_ids


def select_few_shot_examples(
    data_dir: Path,
    n_shots: int,
    exclude_ids: List[str],
    method: str = "m1",
    seed: Optional[int] = None,
) -> List[FewShotExample]:
    """Randomly select few-shot examples from a dataset.
    
    Args:
        data_dir: Dataset directory
        n_shots: Number of examples to select
        exclude_ids: Sample IDs to exclude (e.g., current test sample)
        method: Method name ("m1", "m2", or "m3") - determines what data to load
        seed: Random seed for reproducibility (optional)
        
    Returns:
        List of FewShotExample objects
    """
    if n_shots <= 0:
        return []
    
    # Get all available sample IDs
    all_ids = load_dataset_samples(data_dir)
    
    # Filter out excluded IDs
    available_ids = [sid for sid in all_ids if sid not in exclude_ids]
    
    if len(available_ids) < n_shots:
        logger.warning(
            f"Requested {n_shots} shots but only {len(available_ids)} samples available "
            f"after excluding {len(exclude_ids)} samples. Using all available."
        )
        n_shots = len(available_ids)
    
    # Randomly select sample IDs
    if seed is not None:
        random.seed(seed)
    selected_ids = random.sample(available_ids, n_shots)
    
    # Load data for selected samples
    examples = []
    for sample_id in selected_ids:
        try:
            # Load ground truth (with line breaks - this is the expected output)
            gt_path = data_dir / "gt" / f"{sample_id}.txt"
            gt_text = read_text(gt_path)
            
            # Load transcription (without line breaks - input)
            transcription_path = data_dir / "transcription" / f"{sample_id}.txt"
            transcription = read_text(transcription_path) if transcription_path.exists() else ""
            
            # Load OCR/HTR output (if method requires it)
            ocr_text = None
            if method in ["m2", "m3"]:
                ocr_path = data_dir / "ocr" / f"{sample_id}.txt"
                if ocr_path.exists():
                    ocr_text = read_text(ocr_path)
            
            # Load images (if method requires them)
            image_paths = []
            if method in ["m1", "m2"]:
                images_root = data_dir / "images"
                image_paths = find_images_for_id(images_root, sample_id)
            
            examples.append(FewShotExample(
                sample_id=sample_id,
                gt_text=gt_text,
                transcription=transcription,
                ocr_text=ocr_text,
                image_paths=image_paths,
            ))
            
        except Exception as e:
            logger.warning(f"Failed to load example {sample_id}: {e}")
            continue
    
    return examples

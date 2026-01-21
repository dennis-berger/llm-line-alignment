"""
Script to convert children_handwritten raw data into the standard dataset format.

Input:
- raw_data/csv_aligned/*.csv - CSV files with columns: ID, Category, _0, _1, _2, ...
- raw_data/images/*.png - Image files named by ID

Output:
- gt/{ID}.txt - Ground truth with line breaks (one line per physical line)
- transcription/{ID}.txt - Full text without line breaks
- images/{ID}/ - Folder containing the image for each document
"""

import os
import csv
import shutil
from pathlib import Path


def convert_dataset():
    # Paths
    base_dir = Path(__file__).parent
    raw_data_dir = base_dir / "raw_data"
    csv_dir = raw_data_dir / "csv_aligned"
    images_src_dir = raw_data_dir / "images"
    
    # Output directories
    gt_dir = base_dir / "gt"
    transcription_dir = base_dir / "transcription"
    images_dst_dir = base_dir / "images"
    
    # Create output directories
    gt_dir.mkdir(exist_ok=True)
    transcription_dir.mkdir(exist_ok=True)
    images_dst_dir.mkdir(exist_ok=True)
    
    # Process each CSV file
    csv_files = sorted(csv_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    total_docs = 0
    missing_images = []
    
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file.name}...")
        
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                doc_id = row["ID"]
                
                # Extract text lines (columns starting with _), skip empty ones
                lines = []
                for key in sorted(row.keys()):
                    if key.startswith("_"):
                        value = row[key].strip() if row[key] else ""
                        if value:  # Skip empty cells
                            lines.append(value)
                
                if not lines:
                    print(f"  Warning: No text found for {doc_id}, skipping")
                    continue
                
                # Create gt file (with line breaks)
                gt_path = gt_dir / f"{doc_id}.txt"
                with open(gt_path, "w", encoding="utf-8") as gt_file:
                    gt_file.write("\n".join(lines))
                
                # Create transcription file (no line breaks, space-separated)
                transcription_path = transcription_dir / f"{doc_id}.txt"
                with open(transcription_path, "w", encoding="utf-8") as trans_file:
                    trans_file.write(" ".join(lines))
                
                # Copy image to images/{ID}/ folder
                src_image = images_src_dir / f"{doc_id}.png"
                if src_image.exists():
                    dst_image_dir = images_dst_dir / doc_id
                    dst_image_dir.mkdir(exist_ok=True)
                    dst_image_path = dst_image_dir / f"{doc_id}.png"
                    shutil.copy2(src_image, dst_image_path)
                else:
                    missing_images.append(doc_id)
                    print(f"  Warning: Image not found for {doc_id}")
                
                total_docs += 1
    
    print(f"\n{'='*50}")
    print(f"Conversion complete!")
    print(f"Total documents processed: {total_docs}")
    print(f"Output directories:")
    print(f"  - gt/: {len(list(gt_dir.glob('*.txt')))} files")
    print(f"  - transcription/: {len(list(transcription_dir.glob('*.txt')))} files")
    print(f"  - images/: {len(list(images_dst_dir.iterdir()))} folders")
    
    if missing_images:
        print(f"\nWarning: {len(missing_images)} images not found:")
        for img_id in missing_images:
            print(f"  - {img_id}")


if __name__ == "__main__":
    convert_dataset()

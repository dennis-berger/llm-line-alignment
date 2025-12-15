"""
Convert IAM Original dataset to evaluation format.
Creates two datasets: IAM_handwritten and IAM_print
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional


def get_bounding_box_from_words(line_element) -> Optional[Tuple[int, int, int, int]]:
    """
    Calculate bounding box from all word components in a line.
    Returns (min_x, min_y, max_x, max_y) or None if no components found.
    """
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = 0, 0
    
    found_components = False
    for word in line_element.findall('word'):
        for cmp in word.findall('cmp'):
            found_components = True
            x = int(cmp.get('x'))
            y = int(cmp.get('y'))
            width = int(cmp.get('width'))
            height = int(cmp.get('height'))
            
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + width)
            max_y = max(max_y, y + height)
    
    if not found_components:
        return None
    
    return (min_x, min_y, max_x, max_y)


def get_handwritten_bounding_box(handwritten_part) -> Optional[Tuple[int, int, int, int]]:
    """
    Calculate bounding box for entire handwritten section.
    Returns (min_x, min_y, max_x, max_y) or None if no valid data.
    """
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = 0, 0
    
    found_any = False
    for line in handwritten_part.findall('line'):
        bbox = get_bounding_box_from_words(line)
        if bbox:
            found_any = True
            min_x = min(min_x, bbox[0])
            min_y = min(min_y, bbox[1])
            max_x = max(max_x, bbox[2])
            max_y = max(max_y, bbox[3])
    
    if not found_any:
        return None
    
    # Add some padding
    padding = 10
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = max_x + padding
    max_y = max_y + padding
    
    return (min_x, min_y, max_x, max_y)


def get_machine_printed_bounding_box(form_element, handwritten_bbox: Optional[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    """
    Estimate bounding box for machine-printed section.
    Assumes it's above the handwritten part.
    Skips the header section (Sentence Database title and ID).
    """
    form_width = int(form_element.get('width', 2479))
    
    # Skip the header section at the top (typically ~200-250 pixels)
    # This removes "Sentence Database" and the form ID
    header_skip = 300
    
    # Machine printed text is typically at the top
    # If we have handwritten bbox, use it to determine where machine print ends
    if handwritten_bbox:
        # Machine print is above handwritten, with some margin
        min_y = header_skip
        max_y = handwritten_bbox[1] - 20  # Some margin above handwritten
    else:
        # Fallback: assume machine print is in top ~700 pixels
        min_y = header_skip
        max_y = 700
    
    return (0, min_y, form_width, max_y)


def extract_handwritten_text(handwritten_part) -> List[str]:
    """Extract text lines from handwritten part."""
    lines = []
    for line in handwritten_part.findall('line'):
        text = line.get('text', '').strip()
        if text:
            lines.append(text)
    return lines


def extract_machine_printed_text(machine_part) -> List[str]:
    """Extract text lines from machine-printed part."""
    lines = []
    for line in machine_part.findall('machine-print-line'):
        text = line.get('text', '').strip()
        if text:
            lines.append(text)
    return lines


def process_xml_file(xml_path: Path, forms_dir: Path, output_base: Path, dataset_type: str):
    """
    Process a single XML file and create output files.
    
    Args:
        xml_path: Path to XML file
        forms_dir: Directory containing form images
        output_base: Base directory for output (datasets/)
        dataset_type: 'handwritten' or 'print'
    """
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    form_id = root.get('id')
    if not form_id:
        print(f"Warning: No form ID found in {xml_path.name}")
        return
    
    # Get corresponding image
    image_path = forms_dir / f"{form_id}.png"
    if not image_path.exists():
        print(f"Warning: Image not found for {form_id}: {image_path}")
        return
    
    # Find handwritten and machine-printed parts
    handwritten_part = root.find('handwritten-part')
    machine_part = root.find('machine-printed-part')
    
    # Determine which part to process
    if dataset_type == 'handwritten':
        if handwritten_part is None:
            print(f"Warning: No handwritten part in {form_id}")
            return
        text_lines = extract_handwritten_text(handwritten_part)
        bbox = get_handwritten_bounding_box(handwritten_part)
        output_dir = output_base / 'IAM_handwritten'
    else:  # print
        if machine_part is None:
            print(f"Warning: No machine-printed part in {form_id}")
            return
        text_lines = extract_machine_printed_text(machine_part)
        # Get handwritten bbox to help position machine print
        hw_bbox = get_handwritten_bounding_box(handwritten_part) if handwritten_part else None
        bbox = get_machine_printed_bounding_box(root, hw_bbox)
        output_dir = output_base / 'IAM_print'
    
    if not text_lines:
        print(f"Warning: No text found in {form_id} for {dataset_type}")
        return
    
    # Create output directories
    gt_dir = output_dir / 'gt'
    transcription_dir = output_dir / 'transcription'
    images_dir = output_dir / 'images' / form_id
    
    gt_dir.mkdir(parents=True, exist_ok=True)
    transcription_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Save ground truth (with line breaks)
    gt_text = '\n'.join(text_lines)
    gt_file = gt_dir / f"{form_id}.txt"
    gt_file.write_text(gt_text, encoding='utf-8')
    
    # Save transcription (without line breaks)
    transcription_text = ' '.join(text_lines)
    transcription_file = transcription_dir / f"{form_id}.txt"
    transcription_file.write_text(transcription_text, encoding='utf-8')
    
    # Crop and save image
    if bbox:
        try:
            img = Image.open(image_path)
            cropped_img = img.crop(bbox)
            output_image = images_dir / f"{form_id}.png"
            cropped_img.save(output_image)
            print(f"Processed {form_id} for {dataset_type}: {len(text_lines)} lines")
        except Exception as e:
            print(f"Error processing image for {form_id}: {e}")
    else:
        print(f"Warning: Could not determine bounding box for {form_id}")


def main():
    """Main processing function."""
    # Set up paths
    base_dir = Path(__file__).parent.parent
    iam_dir = base_dir / 'datasets' / 'IAM_original'
    xml_dir = iam_dir / 'xml'
    forms_dir = iam_dir / 'forms'
    output_base = base_dir / 'datasets'
    
    # Check if directories exist
    if not xml_dir.exists():
        print(f"Error: XML directory not found: {xml_dir}")
        return
    
    if not forms_dir.exists():
        print(f"Error: Forms directory not found: {forms_dir}")
        return
    
    # Get all XML files
    xml_files = sorted(xml_dir.glob('*.xml'))
    print(f"Found {len(xml_files)} XML files")
    
    # Process each XML file for both datasets
    print("\n=== Processing IAM_handwritten dataset ===")
    for xml_file in xml_files:
        process_xml_file(xml_file, forms_dir, output_base, 'handwritten')
    
    print("\n=== Processing IAM_print dataset ===")
    for xml_file in xml_files:
        process_xml_file(xml_file, forms_dir, output_base, 'print')
    
    print("\n=== Processing complete ===")
    
    # Print summary
    handwritten_dir = output_base / 'IAM_handwritten'
    print_dir = output_base / 'IAM_print'
    
    if handwritten_dir.exists():
        hw_count = len(list((handwritten_dir / 'gt').glob('*.txt')))
        print(f"IAM_handwritten: {hw_count} files created")
    
    if print_dir.exists():
        pr_count = len(list((print_dir / 'gt').glob('*.txt')))
        print(f"IAM_print: {pr_count} files created")


if __name__ == '__main__':
    main()

"""Prompt templates for different evaluation methods."""

PROMPT_TEMPLATE_M1 = """You see a scanned page with text (either handwritten or printed). You are also given the correct transcription of the entire text. The transcription is accurate but lacks line breaks matching the original page layout.

Your task:
1. Look at the page image carefully and identify where each line of text ends
2. Find where in the transcription the visible text appears
3. Insert line breaks (newlines) at the exact positions where lines end on the page
4. Each line in the image should become one line in your output
5. DO NOT change, add, or remove any characters - only insert newlines
6. Return ONLY the portion of text visible on this page with correct line breaks

{examples}Transcription:
{transcription}

Return the text visible on this page with line breaks matching the image. Each visual line should be on a separate line. Do not add explanations."""

PROMPT_TEMPLATE_M2 = """You see a scanned page with text (either handwritten or printed). You have:
1. The page image showing the actual line breaks
2. The correct transcription (accurate text but no line breaks)
3. HTR/OCR output (may have character errors but shows line breaks)

Your task:
- Align the correct transcription with the HTR/OCR output to find where line breaks should go
- The HTR/OCR output shows the correct LINE STRUCTURE - use this as your primary guide for where to insert line breaks
- Verify the line breaks against the page image when possible
- Insert line breaks into the CORRECT transcription at the positions indicated by the HTR/OCR structure
- Each line in the HTR/OCR should correspond to one line in your output
- DO NOT change any characters in the correct transcription - only insert newlines at line break positions
- Return ONLY the portion of text visible on this page

{examples}Correct transcription:
{transcription}

HTR/OCR output with line breaks:
{htr}

Return the correct transcription with line breaks matching the HTR/OCR structure. Each visual line should be on a separate line. Do not add explanations."""

PROMPT_TEMPLATE_M3 = """You have:
1. A correct transcription of text (accurate but no line breaks)
2. HTR/OCR output of the same text (may have character errors but shows line break positions)

Your task:
- Align the correct transcription with the HTR/OCR output
- Transfer the line break positions from HTR/OCR to the correct transcription
- DO NOT change any characters in the correct transcription - only insert newlines
- Handle cases where HTR/OCR has character errors by finding the best alignment

{examples}Correct transcription:
{transcription}

HTR/OCR output with line breaks:
{htr}

Return the correct transcription with line breaks matching the HTR/OCR structure. Do not add explanations."""


def format_few_shot_examples_m1(examples) -> str:
    """Format few-shot examples for Method 1 (images + transcription).
    
    Args:
        examples: List of FewShotExample objects
        
    Returns:
        Formatted example string to insert into prompt
    """
    if not examples:
        return ""
    
    formatted = "Here are some examples:\n\n"
    
    for i, ex in enumerate(examples, 1):
        formatted += f"Example {i}:\n"
        formatted += f"Transcription:\n{ex.transcription}\n\n"
        formatted += f"Output with line breaks:\n{ex.gt_text}\n\n"
    
    formatted += "Now, apply the same approach to the following:\n\n"
    return formatted


def format_few_shot_examples_m2(examples) -> str:
    """Format few-shot examples for Method 2 (images + transcription + HTR).
    
    Args:
        examples: List of FewShotExample objects
        
    Returns:
        Formatted example string to insert into prompt
    """
    if not examples:
        return ""
    
    formatted = "Here are some examples:\n\n"
    
    for i, ex in enumerate(examples, 1):
        formatted += f"Example {i}:\n"
        formatted += f"Correct transcription:\n{ex.transcription}\n\n"
        formatted += f"HTR/OCR output with line breaks:\n{ex.ocr_text}\n\n"
        formatted += f"Output with correct line breaks:\n{ex.gt_text}\n\n"
    
    formatted += "Now, apply the same approach to the following:\n\n"
    return formatted


def format_few_shot_examples_m3(examples) -> str:
    """Format few-shot examples for Method 3 (transcription + HTR, no images).
    
    Args:
        examples: List of FewShotExample objects
        
    Returns:
        Formatted example string to insert into prompt
    """
    if not examples:
        return ""
    
    formatted = "Here are some examples:\n\n"
    
    for i, ex in enumerate(examples, 1):
        formatted += f"Example {i}:\n"
        formatted += f"Correct transcription:\n{ex.transcription}\n\n"
        formatted += f"HTR/OCR output with line breaks:\n{ex.ocr_text}\n\n"
        formatted += f"Output with correct line breaks:\n{ex.gt_text}\n\n"
    
    formatted += "Now, apply the same approach to the following:\n\n"
    return formatted

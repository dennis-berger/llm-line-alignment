"""Prompt templates for different evaluation methods."""

PROMPT_TEMPLATE_M1 = """# Role and Objective
Process a scanned page image and its transcription (continuous text; no line breaks) to reconstruct line breaks that match the page's visual line layout.

# Instructions
- Examine the page image to locate each visually distinct line of text, including headings, marginalia, and indented lines.
- Locate the exact wording for each visible line within the transcription.
- Insert newline characters at precise locations matching the visual end of each line in the image.
- Do not add, remove, or alter any characters other than adding newlines.
- Output only the visible text from the provided page, segmented precisely by these line breaks.
- Ignore image lines that do not have an exact transcriptional match.

# Output Format
Return a single block of text: each line corresponds to a visually distinct line in the image, separated by newline characters (not '\\n'). Do not include explanations, errors, or extra formatting—output only the corrected text segment.

{examples}Transcription:
{transcription}"""

PROMPT_TEMPLATE_M2 = """# Role and Objective
Align the correct transcription of a scanned page to match the visual line structure, using the page image as the primary reference and HTR/OCR output as a structural guide.

# Instructions
- Use the page image as the main reference for true visual line breaks.
- Use the HTR/OCR output to help identify line boundaries, especially where the image is ambiguous.
- Insert newline characters into the correct transcription at positions matching visual line ends.
- Do not add, remove, or alter any characters other than adding newlines.
- Each visual line in the image should correspond to one line in your output.
- Output only the visible text from this page, segmented by line breaks.

# Output Format
Return the correct transcription with inserted line breaks. Each line corresponds to a visual line in the image. Do not include explanations or extra formatting.

{examples}Correct transcription:
{transcription}

HTR/OCR output with line breaks:
{htr}"""

PROMPT_TEMPLATE_M3 = """# Role and Objective
Align a correct transcription (no line breaks) to match the line structure indicated by an HTR/OCR output (which has line breaks but may contain character errors).

# Instructions
- Use the HTR/OCR output to determine where line breaks should be inserted in the correct transcription.
- Do not add, remove, or alter any characters in the transcription other than inserting newlines.
- Handle HTR/OCR character errors by finding the best alignment between corresponding text segments.
- Each line in the HTR/OCR should correspond to one line in your output.

# Output Format
Return only the correct transcription with inserted line breaks matching the HTR/OCR structure. Do not include explanations, code blocks, or extra formatting.

{examples}Correct transcription:
{transcription}

HTR/OCR output with line breaks:
{htr}"""


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

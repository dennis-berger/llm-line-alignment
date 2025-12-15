"""Prompt templates for different evaluation methods."""

PROMPT_TEMPLATE_M1 = """You see a scanned page with text (either handwritten or printed). You are also given the correct transcription of the entire text. The transcription is accurate but lacks line breaks matching the original page layout.

Your task:
1. Look at the page image carefully and identify where each line of text ends
2. Find where in the transcription the visible text appears
3. Insert line breaks (newlines) at the exact positions where lines end on the page
4. Each line in the image should become one line in your output
5. DO NOT change, add, or remove any characters - only insert newlines
6. Return ONLY the portion of text visible on this page with correct line breaks

Transcription:
{transcription}

Return the text visible on this page with line breaks matching the image. Each visual line should be on a separate line. Do not add explanations."""

PROMPT_TEMPLATE_M2 = """You see a scanned page with text (either handwritten or printed). You have:
1. The page image showing the actual line breaks
2. The correct transcription (accurate text but no line breaks)
3. HTR/OCR output (may have character errors but shows approximate line breaks)

Your task:
- Look at the image to see where each line of text ends
- Use the HTR/OCR line breaks as hints for alignment
- Insert line breaks into the CORRECT transcription at positions matching the visual lines on the page
- Each visual line in the image should become one line in your output
- DO NOT change any characters in the correct transcription - only insert newlines
- Return ONLY the portion of text visible on this page

Correct transcription:
{transcription}

HTR/OCR output with line breaks:
{htr}

Return the text visible on this page with correct line breaks matching the image. Each visual line should be on a separate line. Do not add explanations."""

PROMPT_TEMPLATE_M3 = """You have:
1. A correct transcription of text (accurate but no line breaks)
2. HTR/OCR output of the same text (may have character errors but shows line break positions)

Your task:
- Align the correct transcription with the HTR/OCR output
- Transfer the line break positions from HTR/OCR to the correct transcription
- DO NOT change any characters in the correct transcription - only insert newlines
- Handle cases where HTR/OCR has character errors by finding the best alignment

Correct transcription:
{transcription}

HTR/OCR output with line breaks:
{htr}

Return the correct transcription with line breaks matching the HTR/OCR structure. Do not add explanations."""

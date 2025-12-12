"""Prompt templates for different evaluation methods."""

PROMPT_TEMPLATE_M1 = """You see a scanned page from a historical handwritten letter. You are also given the correct diplomatic transcription of the entire letter. The transcription is accurate but lacks line breaks matching the original page layout.

Your task:
1. Look at the page image carefully
2. Find where in the transcription the visible text appears
3. Insert line breaks (newlines) at the exact positions where lines end on the page
4. DO NOT change, add, or remove any characters - only insert newlines
5. Return ONLY the portion of text visible on this page with correct line breaks

Transcription:
{transcription}

Return the text visible on this page with line breaks matching the image. Do not add explanations."""

PROMPT_TEMPLATE_M2 = """You see a scanned page from a historical handwritten letter. You have:
1. The page image
2. The correct diplomatic transcription (accurate text but no line breaks)
3. HTR output (may have errors but shows approximate line breaks)

Your task:
- Use the image to see where lines actually break
- Use the HTR line breaks as hints
- Insert line breaks into the CORRECT transcription at positions matching the page
- DO NOT change any characters in the correct transcription - only insert newlines
- Return ONLY the portion of text visible on this page

Correct transcription:
{transcription}

HTR output with line breaks:
{htr}

Return the text visible on this page with correct line breaks. Do not add explanations."""

PROMPT_TEMPLATE_M3 = """You have:
1. A correct diplomatic transcription of a handwritten letter (accurate but no line breaks)
2. HTR output of the same text (may have character errors but shows line break positions)

Your task:
- Align the correct transcription with the HTR output
- Transfer the line break positions from HTR to the correct transcription
- DO NOT change any characters in the correct transcription - only insert newlines
- Handle cases where HTR has character errors by finding the best alignment

Correct transcription:
{transcription}

HTR output with line breaks:
{htr}

Return the correct transcription with line breaks matching the HTR structure. Do not add explanations."""

"""Prompt templates for different evaluation methods."""

import json

from utils.m4 import render_ocr_line_hints

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

M4_PROMPT_INSTRUCTION_BLOCKS = {
    "baseline": """- The transcription is the only source of characters.
- The PyLaia line hypotheses are structural hints only; they may contain OCR errors.
- Produce exactly {num_lines} output lines in reading order.
- Do not add, remove, or alter transcription characters.
- Do not copy page or line metadata into the output.
- The concatenation of your output lines must exactly equal the transcription.""",
    "boundary_anchored_v1": """- The transcription is the only source of characters.
- The PyLaia line hypotheses are structural hints only; they may contain OCR errors.
- Produce exactly {num_lines} output lines in reading order.
- Treat PyLaia hint line i as the anchor for output line i.
- Do not reflow the text into cleaner, more semantic, or more balanced lines than the hints suggest.
- Preserve short, odd, or fragmentary standalone lines when the hints suggest them, including headers, dates, page numbers, symbols, and brief tail fragments.
- When uncertain between two nearby split points, prefer the split that minimizes boundary drift into neighboring lines.
- Do not add, remove, or alter transcription characters.
- Do not copy page or line metadata into the output.
- The concatenation of your output lines must exactly equal the transcription.""",
}

PROMPT_TEMPLATE_M4 = """# Role and Objective
Align a correct transcription (no line breaks) to an ordered list of PyLaia line hypotheses.

# Instructions
{instruction_block}

# Output Format
Return strict JSON only, with no code fences or extra text:
{{"lines": ["...", "..."]}}

{examples}Correct transcription:
{transcription}

Ordered PyLaia line hypotheses ({num_lines} lines):
{line_hints}"""

PROMPT_TEMPLATE_M4_BASELINE = PROMPT_TEMPLATE_M4.replace(
    "{instruction_block}",
    M4_PROMPT_INSTRUCTION_BLOCKS["baseline"],
)
PROMPT_TEMPLATE_M4_BOUNDARY_ANCHORED_V1 = PROMPT_TEMPLATE_M4.replace(
    "{instruction_block}",
    M4_PROMPT_INSTRUCTION_BLOCKS["boundary_anchored_v1"],
)
M4_PROMPT_TEMPLATES = {
    "baseline": PROMPT_TEMPLATE_M4_BASELINE,
    "boundary_anchored_v1": PROMPT_TEMPLATE_M4_BOUNDARY_ANCHORED_V1,
}
M4_PROMPT_VARIANTS = tuple(M4_PROMPT_TEMPLATES)
PROMPT_TEMPLATE_M4 = PROMPT_TEMPLATE_M4_BASELINE

M4_REPAIR_PROMPT_TEMPLATE = """Your previous response was invalid.
Error: {error_message}

Follow these rules:
{instruction_block}
- Return only strict JSON with exactly {num_lines} strings in the "lines" array.

Correct transcription:
{transcription}

Ordered PyLaia line hypotheses ({num_lines} lines):
{line_hints}

Previous invalid response:
{previous_response}"""


def get_m4_prompt_template(variant: str = "baseline") -> str:
    """Return the full M4 prompt template for one variant."""

    try:
        return M4_PROMPT_TEMPLATES[variant]
    except KeyError as exc:
        raise ValueError(
            f"Unknown M4 prompt variant '{variant}'. Expected one of {', '.join(M4_PROMPT_VARIANTS)}."
        ) from exc


def build_m4_repair_prompt(
    *,
    variant: str,
    error_message: str,
    num_lines: int,
    transcription: str,
    line_hints: str,
    previous_response: str,
) -> str:
    """Build the M4 repair prompt for one variant."""

    try:
        instruction_block = M4_PROMPT_INSTRUCTION_BLOCKS[variant].format(num_lines=num_lines)
    except KeyError as exc:
        raise ValueError(
            f"Unknown M4 prompt variant '{variant}'. Expected one of {', '.join(M4_PROMPT_VARIANTS)}."
        ) from exc

    return M4_REPAIR_PROMPT_TEMPLATE.format(
        error_message=error_message,
        instruction_block=instruction_block,
        num_lines=num_lines,
        transcription=transcription,
        line_hints=line_hints,
        previous_response=previous_response,
    )

M5_PROMPT_INSTRUCTION_BLOCKS = {
    "baseline": """- The transcription is the only source of characters.
- Use the supplied line images to infer where each transcription line should begin and end.
- Produce exactly {num_lines} output lines in the same reading order as the supplied line images.
- Do not add, remove, or alter transcription characters.
- Do not copy metadata into the output.
- The concatenation of your output lines must exactly equal the transcription.
- {image_mode_description}
- {page_image_instruction}
{ocr_text_instruction}""",
    "boundary_anchored_v1": """- The transcription is the only source of characters.
- Use the supplied line images to infer where each transcription line should begin and end.
- Produce exactly {num_lines} output lines in the same reading order as the supplied line images.
- Treat line image i as the anchor for output line i.
- Do not reflow the text into cleaner, more semantic, or more balanced lines than the images suggest.
- Preserve short, odd, or fragmentary standalone lines when the images suggest them, including page numbers, dates, headings, salutations, symbols, and brief tail fragments.
- If a line image appears shorter than its neighbors, prefer a shorter transcription span for that line instead of absorbing it into an adjacent line.
- When uncertain between two nearby split points, prefer the earlier split that avoids pushing text into later lines.
- Do not leave the first or last output line empty unless the corresponding line image is genuinely blank.
- Do not add, remove, or alter transcription characters.
- Do not copy metadata into the output.
- The concatenation of your output lines must exactly equal the transcription.
- {image_mode_description}
- {page_image_instruction}
{ocr_text_instruction}""",
    "boundary_context_v2": """- The transcription is the only source of characters.
- Use the supplied line-level images as the primary anchors for line boundaries.
- Produce exactly {num_lines} output lines in the same reading order as the supplied line images.
- Treat line image i as the hard local anchor for output line i.
- Keep uncertainty local: if one line is ambiguous, do not absorb text from its neighbors unless the images clearly require it.
- Preserve short, odd, or fragmentary standalone lines when the images suggest them, including page numbers, dates, headings, salutations, symbols, addresses, and brief tail fragments.
- If a line image appears much shorter than its neighbors, prefer assigning it a shorter transcription span instead of smoothing the boundary away.
- Use the global page context only to prevent long-range drift, not to override a clear local line image anchor.
- When uncertain between two nearby split points, prefer the earlier split that avoids pushing text into later lines.
- Do not leave the first or last output line empty unless the corresponding line image is genuinely blank.
- Do not add, remove, or alter transcription characters.
- Do not copy metadata into the output.
- The concatenation of your output lines must exactly equal the transcription.
- {image_mode_description}
- {page_image_instruction}
{ocr_text_instruction}""",
}

PROMPT_TEMPLATE_M5 = """# Role and Objective
Align a correct transcription (no line breaks) to an ordered set of line images.

# Instructions
{instruction_block}

# Output Format
Return strict JSON only, with no code fences or extra text:
{{"lines": ["...", "..."]}}

{examples}Correct transcription:
{transcription}

Ordered line-image manifest ({num_lines} lines):
{line_image_manifest}
{ocr_text_section}"""

PROMPT_TEMPLATE_M5_BASELINE = PROMPT_TEMPLATE_M5.replace(
    "{instruction_block}",
    M5_PROMPT_INSTRUCTION_BLOCKS["baseline"],
)
PROMPT_TEMPLATE_M5_BOUNDARY_ANCHORED_V1 = PROMPT_TEMPLATE_M5.replace(
    "{instruction_block}",
    M5_PROMPT_INSTRUCTION_BLOCKS["boundary_anchored_v1"],
)
PROMPT_TEMPLATE_M5_BOUNDARY_CONTEXT_V2 = PROMPT_TEMPLATE_M5.replace(
    "{instruction_block}",
    M5_PROMPT_INSTRUCTION_BLOCKS["boundary_context_v2"],
)
M5_PROMPT_TEMPLATES = {
    "baseline": PROMPT_TEMPLATE_M5_BASELINE,
    "boundary_anchored_v1": PROMPT_TEMPLATE_M5_BOUNDARY_ANCHORED_V1,
    "boundary_context_v2": PROMPT_TEMPLATE_M5_BOUNDARY_CONTEXT_V2,
}
M5_PROMPT_VARIANTS = tuple(M5_PROMPT_TEMPLATES)
PROMPT_TEMPLATE_M5 = PROMPT_TEMPLATE_M5_BASELINE

M5_REPAIR_PROMPT_TEMPLATE = """Your previous response was invalid.
Error: {error_message}

Follow these rules:
{instruction_block}
- Return only strict JSON with exactly {num_lines} strings in the "lines" array.

Correct transcription:
{transcription}

Ordered line-image manifest ({num_lines} lines):
{line_image_manifest}
{ocr_text_section}

Previous invalid response:
{previous_response}"""

M5_CANDIDATE_JUDGE_PROMPT_TEMPLATE = """# Role and Objective
Choose the better of two candidate line segmentations for the same transcription by comparing them against the supplied images.

# Instructions
- The transcription is the only source of characters.
- Candidate A was generated as: {candidate_a_description}
- Candidate B was generated as: {candidate_b_description}
- Use the supplied images to decide which candidate matches the visual line boundaries better overall.
- Prefer the candidate that preserves short standalone lines, signatures, dates, headings, addresses, and other visually short fragments.
- Prefer the candidate that avoids one bad split cascading into many later lines.
- Do not invent a third segmentation. Choose the better existing candidate.
- Return strict JSON only, with no code fences or extra text:
{{"winner": "A", "reason": "short explanation"}}

Correct transcription:
{transcription}

Ordered line-image manifest ({num_lines} lines):
{line_image_manifest}
{ocr_text_section}

Candidate A ({candidate_a_description}):
{candidate_a_lines}

Candidate B ({candidate_b_description}):
{candidate_b_lines}"""

M5_CANDIDATE_JUDGE_REPAIR_PROMPT_TEMPLATE = """Your previous response was invalid.
Error: {error_message}

Follow these rules:
- Choose exactly one winner: "A" or "B".
- Return only strict JSON:
{{"winner": "A", "reason": "short explanation"}}

Correct transcription:
{transcription}

Ordered line-image manifest ({num_lines} lines):
{line_image_manifest}
{ocr_text_section}

Candidate A ({candidate_a_description}):
{candidate_a_lines}

Candidate B ({candidate_b_description}):
{candidate_b_lines}

Previous invalid response:
{previous_response}"""


def get_m5_prompt_template(variant: str = "baseline") -> str:
    """Return the full M5 prompt template for one variant."""

    try:
        return M5_PROMPT_TEMPLATES[variant]
    except KeyError as exc:
        raise ValueError(
            f"Unknown M5 prompt variant '{variant}'. Expected one of {', '.join(M5_PROMPT_VARIANTS)}."
        ) from exc


def build_m5_repair_prompt(
    *,
    variant: str,
    error_message: str,
    num_lines: int,
    transcription: str,
    line_image_manifest: str,
    image_mode_description: str,
    page_image_instruction: str,
    ocr_text_instruction: str,
    ocr_text_section: str,
    previous_response: str,
) -> str:
    """Build the M5 repair prompt for one variant."""

    try:
        instruction_block = M5_PROMPT_INSTRUCTION_BLOCKS[variant].format(
            num_lines=num_lines,
            image_mode_description=image_mode_description,
            page_image_instruction=page_image_instruction,
            ocr_text_instruction=ocr_text_instruction,
        )
    except KeyError as exc:
        raise ValueError(
            f"Unknown M5 prompt variant '{variant}'. Expected one of {', '.join(M5_PROMPT_VARIANTS)}."
        ) from exc

    return M5_REPAIR_PROMPT_TEMPLATE.format(
        error_message=error_message,
        instruction_block=instruction_block,
        num_lines=num_lines,
        transcription=transcription,
        line_image_manifest=line_image_manifest,
        ocr_text_section=ocr_text_section,
        previous_response=previous_response,
    )


def render_m5_candidate_lines(lines: list[str]) -> str:
    """Render one candidate segmentation as a numbered list for the judge prompt."""

    return "\n".join(
        f"{index + 1}. {line}"
        for index, line in enumerate(lines)
    )


def build_m5_candidate_judge_prompt(
    *,
    transcription: str,
    num_lines: int,
    line_image_manifest: str,
    ocr_text_section: str,
    candidate_a_description: str,
    candidate_b_description: str,
    candidate_a_lines: list[str],
    candidate_b_lines: list[str],
) -> str:
    """Build the candidate-comparison prompt for the M5 judge pass."""

    return M5_CANDIDATE_JUDGE_PROMPT_TEMPLATE.format(
        transcription=transcription,
        num_lines=num_lines,
        line_image_manifest=line_image_manifest,
        ocr_text_section=ocr_text_section,
        candidate_a_description=candidate_a_description,
        candidate_b_description=candidate_b_description,
        candidate_a_lines=render_m5_candidate_lines(candidate_a_lines),
        candidate_b_lines=render_m5_candidate_lines(candidate_b_lines),
    )


def build_m5_candidate_judge_repair_prompt(
    *,
    error_message: str,
    transcription: str,
    num_lines: int,
    line_image_manifest: str,
    ocr_text_section: str,
    candidate_a_description: str,
    candidate_b_description: str,
    candidate_a_lines: list[str],
    candidate_b_lines: list[str],
    previous_response: str,
) -> str:
    """Build the repair prompt for an invalid M5 judge response."""

    return M5_CANDIDATE_JUDGE_REPAIR_PROMPT_TEMPLATE.format(
        error_message=error_message,
        transcription=transcription,
        num_lines=num_lines,
        line_image_manifest=line_image_manifest,
        ocr_text_section=ocr_text_section,
        candidate_a_description=candidate_a_description,
        candidate_b_description=candidate_b_description,
        candidate_a_lines=render_m5_candidate_lines(candidate_a_lines),
        candidate_b_lines=render_m5_candidate_lines(candidate_b_lines),
        previous_response=previous_response,
    )


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


def format_few_shot_examples_m4(examples) -> str:
    """Format few-shot examples for Method 4 (transcription + structured PyLaia lines)."""
    if not examples:
        return ""

    formatted = "Here are some examples:\n\n"

    for i, ex in enumerate(examples, 1):
        if not ex.ocr_lines:
            continue
        formatted += f"Example {i}:\n"
        formatted += f"Correct transcription:\n{ex.transcription}\n\n"
        formatted += "Ordered PyLaia line hypotheses:\n"
        formatted += f"{render_ocr_line_hints(ex.ocr_lines['lines'])}\n\n"
        formatted += "Output JSON:\n"
        formatted += (
            json.dumps({"lines": ex.gt_text.splitlines()}, ensure_ascii=False)
            + "\n\n"
        )

    formatted += "Now, apply the same approach to the following:\n\n"
    return formatted


def format_few_shot_examples_m5(examples, use_ocr_text: bool = False) -> str:
    """Format few-shot examples for Method 5 (transcription + line images)."""
    if not examples:
        return ""

    formatted = (
        "Here are some examples. Their example line images are supplied before the target images, "
        "and each example's images are already in reading order.\n\n"
    )

    for i, ex in enumerate(examples, 1):
        if not ex.line_image_paths:
            continue
        formatted += f"Example {i}:\n"
        formatted += f"Correct transcription:\n{ex.transcription}\n\n"
        formatted += (
            "Ordered line-image manifest:\n"
            + "\n".join(
                f"{index + 1}. example line image {index + 1}"
                for index in range(len(ex.line_image_paths))
            )
            + "\n\n"
        )
        if use_ocr_text and ex.ocr_lines:
            formatted += "Optional OCR line hints:\n"
            formatted += f"{render_ocr_line_hints(ex.ocr_lines['lines'])}\n\n"
        formatted += "Output JSON:\n"
        formatted += (
            json.dumps({"lines": ex.gt_text.splitlines()}, ensure_ascii=False)
            + "\n\n"
        )

    formatted += "Now, apply the same approach to the following target sample:\n\n"
    return formatted

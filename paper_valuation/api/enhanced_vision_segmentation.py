# enhanced_vision_segmentation.py
# IMPROVED VERSION — Separate short/long reconstruction, robust paragraph detection

from google.cloud import vision
import re
from typing import Dict, List, Optional

BULLET_MARKERS = ['•', '●', '○', '-', '*', '→', '▸', '>', '■', '□', '▪', '◆', '◇', '►', '»', '–', '—']


# ─────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────

def calculate_average_line_height(words: List[Dict]) -> float:
    """
    Calculate median line height (more robust than mean for handwriting).
    Filters out extremely small or large values to reduce noise.
    """
    heights = [w['max_y'] - w['y'] for w in words if (w['max_y'] - w['y']) > 5]
    if not heights:
        return 40
    heights.sort()
    # Use median instead of mean — handwriting has outliers
    mid = len(heights) // 2
    return heights[mid]


def calculate_dominant_x_position(words: List[Dict]) -> float:
    """
    Find the most common left-margin X position using 10px bucket grouping.
    """
    x_groups: Dict[float, List[float]] = {}
    for word in words:
        x = word['x']
        placed = False
        for key in x_groups:
            if abs(x - key) < 10:
                x_groups[key].append(x)
                placed = True
                break
        if not placed:
            x_groups[x] = [x]

    if not x_groups:
        return 0
    largest = max(x_groups.values(), key=len)
    return sum(largest) / len(largest)


# ─────────────────────────────────────────────
# SHORT ANSWER RECONSTRUCTION
# Simple flat concatenation — no paragraph logic needed
# ─────────────────────────────────────────────

def reconstruct_short_answer(words: List[Dict], start_idx: int, end_idx: Optional[int] = None) -> str:
    """
    For short answers: join all words into a single clean string.
    Ignores Y-gaps completely — a short answer is always one semantic unit.
    Only respects SPACE break types; everything else gets a space.
    """
    if end_idx is None:
        end_idx = len(words)

    parts = []
    for i in range(start_idx, min(end_idx, len(words))):
        word = words[i]
        parts.append(word['text'])
        if i < end_idx - 1:
            # Always add a space between words in short answers
            parts.append(' ')

    text = ''.join(parts)
    # Collapse multiple spaces, strip
    text = re.sub(r' +', ' ', text)
    return text.strip()


# ─────────────────────────────────────────────
# LONG ANSWER RECONSTRUCTION
# Paragraph-aware, adaptive threshold detection
# ─────────────────────────────────────────────

def detect_paragraph_boundary(
    current_word: Dict,
    next_word: Dict,
    avg_line_height: float,
    dominant_x: float
) -> bool:
    """
    Decide whether there is a paragraph break AFTER current_word.

    Rules (any ONE sufficient condition triggers a break):
    1. Y-gap > 1.5× avg line height  (generous gap — clear visual separation)
    2. Next word is a bullet marker
    3. Y-gap > 1.0× AND sentence ended with punctuation
    4. Y-gap > 1.0× AND next word starts a new indented block (x shift > 30px from dominant)

    Using 1.5× instead of 1.3× reduces false positives from normal
    handwriting line spacing variation.
    """
    y_gap = next_word['y'] - current_word['max_y']

    # Ignore negative gaps (words on the same line detected out of order)
    if y_gap < 0:
        return False

    is_line_break = current_word['break_type'] in [
        vision.TextAnnotation.DetectedBreak.BreakType.LINE_BREAK,
        vision.TextAnnotation.DetectedBreak.BreakType.EOL_SURE_SPACE,
    ]

    if not is_line_break:
        return False  # Same line — never a paragraph break

    is_bullet = (
        next_word['text'].strip() in BULLET_MARKERS
        or (len(next_word['text']) > 0 and next_word['text'][0] in BULLET_MARKERS)
    )
    sentence_end = current_word['text'].rstrip().endswith(('.', '!', '?', ':'))
    large_gap = y_gap > (avg_line_height * 1.5)
    moderate_gap = y_gap > (avg_line_height * 1.0)
    new_indent = abs(next_word['x'] - dominant_x) > 30

    if large_gap:
        return True
    if is_bullet:
        return True
    if moderate_gap and sentence_end:
        return True
    if moderate_gap and new_indent:
        return True

    return False


def reconstruct_long_answer(
    words: List[Dict],
    start_idx: int,
    end_idx: Optional[int] = None,
    is_handwritten: bool = True
) -> str:
    """
    For long answers: reconstruct with paragraph breaks.
    Uses adaptive thresholds based on the actual document's line height.
    """
    if end_idx is None:
        end_idx = len(words)

    answer_words = words[start_idx:end_idx]
    if not answer_words:
        return ''

    avg_line_height = calculate_average_line_height(answer_words)
    dominant_x = calculate_dominant_x_position(answer_words)

    parts = []
    paragraph_breaks = 0

    for i in range(start_idx, min(end_idx, len(words))):
        word = words[i]
        parts.append(word['text'])

        if i >= end_idx - 1:
            continue

        next_word = words[i + 1]

        if detect_paragraph_boundary(word, next_word, avg_line_height, dominant_x):
            parts.append('\n\n')
            paragraph_breaks += 1
        elif word['break_type'] in [
            vision.TextAnnotation.DetectedBreak.BreakType.LINE_BREAK,
            vision.TextAnnotation.DetectedBreak.BreakType.EOL_SURE_SPACE,
        ]:
            # Line break but not a paragraph — just a space
            parts.append(' ')
        elif word['has_space_after']:
            parts.append(' ')

    text = ''.join(parts)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +\n', '\n', text)
    text = re.sub(r'\n +', '\n', text)

    print(f"      📊 Long answer: {paragraph_breaks} paragraph breaks, avg line height {avg_line_height:.1f}px")

    return text.strip()


# ─────────────────────────────────────────────
# UNIFIED ENTRY POINT
# Called by vision_segmentation.py for all answers
# ─────────────────────────────────────────────

def reconstruct_answer_text_adaptive(
    words: List[Dict],
    start_idx: int,
    end_idx: Optional[int] = None,
    is_handwritten: bool = True,
    answer_type: str = 'short'   # ← NEW PARAMETER
) -> str:
    """
    Unified reconstruction dispatcher.

    Args:
        words:        Full word list from OCR
        start_idx:    Start index for this answer's words
        end_idx:      End index (exclusive)
        is_handwritten: Affects threshold scaling
        answer_type:  'short' or 'long' — controls which algorithm is used

    Short answers → flat string, no paragraph logic
    Long answers  → adaptive paragraph detection
    """
    if answer_type == 'long':
        return reconstruct_long_answer(words, start_idx, end_idx, is_handwritten)
    else:
        return reconstruct_short_answer(words, start_idx, end_idx)


# ─────────────────────────────────────────────
# VALIDATION HELPERS (unchanged from original)
# ─────────────────────────────────────────────

def validate_answer_format(document_annotation, strict_mode: bool = False,
                           expected_questions: List[int] = None) -> tuple:
    from paper_valuation.api.vision_segmentation import (
        extract_word_level_data,
        find_all_question_labels,
    )

    word_data = extract_word_level_data(document_annotation)
    boundaries = find_all_question_labels(word_data)

    errors = []
    warnings = []

    if not boundaries:
        errors.append("No question labels found! Please write Q1, Q2, etc. on the left side.")
        return False, errors, warnings

    question_numbers = [b['q_number'] for b in boundaries]

    if len(question_numbers) != len(set(question_numbers)):
        duplicates = [q for q in set(question_numbers) if question_numbers.count(q) > 1]
        errors.append(f"Duplicate question numbers: {', '.join([f'Q{d}' for d in duplicates])}")

    if question_numbers != sorted(question_numbers):
        warnings.append(f"Questions out of order: {' → '.join([f'Q{q}' for q in question_numbers])}")

    for i in range(len(boundaries) - 1):
        spacing = boundaries[i + 1]['y_start'] - boundaries[i]['y_start']
        if spacing < 50:
            warnings.append(
                f"Q{boundaries[i]['q_number']} and Q{boundaries[i+1]['q_number']} are too close."
            )
        if strict_mode and spacing < 100:
            errors.append(
                f"Strict mode: Q{boundaries[i]['q_number']} and Q{boundaries[i+1]['q_number']} "
                f"must have at least 3 lines of spacing."
            )

    return len(errors) == 0, errors, warnings


def analyze_answer_structure(text: str) -> Dict:
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    bullet_count = sum(1 for p in paragraphs if any(p.startswith(b) for b in BULLET_MARKERS))
    return {
        'paragraph_count': len(paragraphs),
        'bullet_point_count': bullet_count,
        'avg_paragraph_length': sum(len(p) for p in paragraphs) / len(paragraphs) if paragraphs else 0,
        'total_length': len(text),
        'has_bullets': bullet_count > 0,
        'is_structured': bullet_count >= 2,
    }


def normalize_bullet_points(text: str) -> str:
    """Convert various bullet markers to standard • format."""
    for bullet in BULLET_MARKERS:
        if bullet != '•':
            text = text.replace(bullet, '•')
    return text


def count_paragraph_breaks(text: str) -> int:
    """Count double-newline paragraph breaks."""
    return text.count('\n\n')


def generate_formatting_suggestions(errors: List[str], warnings: List[str]) -> List[str]:
    """Generate helpful suggestions based on validation results."""
    suggestions = []
    if any("No question labels" in e for e in errors):
        suggestions.append("Write 'Q1', 'Q2', etc. at the START of each answer on the LEFT side")
    if any("Duplicate" in e for e in errors):
        suggestions.append("Check your question numbers — each question should appear only once")
    if any("too close" in w for w in warnings):
        suggestions.append("Leave MORE SPACE between questions (at least 2-3 blank lines)")
    if any("out of order" in w for w in warnings):
        suggestions.append("Write questions in order: Q1, Q2, Q3... (or clearly mark continuation pages)")
    if not suggestions:
        suggestions.append("Format looks good! Remember to use bullet points for long answers.")
    return suggestions


# ─────────────────────────────────────────────────────────────
# INTEGRATION HELPER
# Drop-in replacement for the original segment_answers()
# Usage:
#   from enhanced_vision_segmentation import segment_answers_enhanced as segment_answers
# ─────────────────────────────────────────────────────────────

def segment_answers_enhanced(document_annotation, debug=True, config=None):
    """
    Enhanced wrapper using adaptive reconstruction.

    DROP-IN REPLACEMENT for vision_segmentation.segment_answers().
    Used by the heuristic fallback path in vision_segmentation.py.

    Differences from original segment_answers():
    - Short answers use flat reconstruction (no paragraph logic)
    - Long answers use adaptive threshold based on actual document line height
    - Strict Q+number label matching only
    - Same-Q label on one page = continuation (concatenated, not error)
    """
    from paper_valuation.api.vision_segmentation import (
        extract_word_level_data,
        find_all_question_labels,
        validate_question_sequence,
        is_question_label,
        clean_answer_text,
    )

    if config is None:
        config = {}

    is_handwritten = config.get('is_handwritten', True)
    question_types = config.get('question_types', {})
    default_answer_type = config.get('default_answer_type', 'short')

    word_data = extract_word_level_data(document_annotation)

    if debug:
        print(f"\n{'='*70}")
        print(f"📄 ENHANCED ADAPTIVE SEGMENTATION")
        print(f"   Words extracted: {len(word_data)}")
        print(f"   Document type: {'Handwritten' if is_handwritten else 'Printed'}")
        print(f"{'='*70}")

    boundaries = find_all_question_labels(
        word_data,
        config.get('left_margin_threshold', 400),
        config.get('max_expected_question', 20)
    )

    if debug:
        print(f"\n🔍 Found {len(boundaries)} question labels")
        for b in boundaries:
            q_type = question_types.get(str(b['q_number']), default_answer_type)
            print(f"   • {b['label']} at Y={b['y_start']}, Type: {q_type}")

    is_valid, missing, warnings, validation_info = validate_question_sequence(
        boundaries,
        strict=config.get('strict_validation', False),
        expected_questions=config.get('expected_questions', None)
    )

    if debug:
        print(f"\n✓ VALIDATION:")
        for warning in warnings:
            print(f"   {warning}")

    segmented_answers = {}

    for i, boundary in enumerate(boundaries):
        start_idx = boundary['word_index']
        if start_idx < len(word_data) and is_question_label(word_data[start_idx]['text']) is not None:
            start_idx += 1
        end_idx = boundaries[i + 1]['word_index'] if i + 1 < len(boundaries) else len(word_data)

        q_num = boundary['q_number']
        q_label = boundary['label']
        answer_type = question_types.get(str(q_num), default_answer_type)

        if debug:
            print(f"\n   Processing {q_label} ({answer_type} answer)...")

        answer_text = reconstruct_answer_text_adaptive(
            word_data, start_idx, end_idx,
            is_handwritten=is_handwritten,
            answer_type=answer_type
        )
        answer_text = clean_answer_text(answer_text, q_num)

        structure = analyze_answer_structure(answer_text)
        if debug:
            print(f"      Structure: {structure['paragraph_count']} paragraphs, "
                  f"{structure['bullet_point_count']} bullets")

        # Same-Q label on same page = continuation, concatenate
        if q_label in segmented_answers:
            if answer_text:
                segmented_answers[q_label] = segmented_answers[q_label] + ' ' + answer_text
        else:
            segmented_answers[q_label] = answer_text

    if debug:
        print(f"\n{'='*70}")
        print(f"✅ Extracted {len(segmented_answers)} answers")
        print(f"{'='*70}\n")

    return {
        'answers': segmented_answers,
        'metadata': {
            'total_questions_found': len(boundaries),
            'question_numbers': validation_info.get('found_questions', []),
            'is_complete': is_valid,
            'missing_questions': missing,
            'is_handwritten': is_handwritten,
            'detection_method': 'adaptive_multi_signal'
        },
        'validation': {
            'is_valid': is_valid,
            'warnings': warnings,
            'info': validation_info
        }
    }

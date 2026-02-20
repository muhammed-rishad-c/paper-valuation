from google.cloud import vision
import google.auth
import io
import os
import re
from typing import Dict, List, Optional, Tuple
from paper_valuation.logging.logger import logging

from dotenv import load_dotenv
load_dotenv()

_SERVICE_ACCOUNT_KEY_FILE = os.environ.get("SERVICE_ACCOUNT_KEY_FILE")

if not _SERVICE_ACCOUNT_KEY_FILE:
    raise ValueError("SERVICE_ACCOUNT_KEY_FILE not found. Check if .env is loaded correctly.")


# ─────────────────────────────────────────────────────────────
# OCR CLIENT
# ─────────────────────────────────────────────────────────────

def get_document_annotation(image_path: str):
    credentials, _ = google.auth.load_credentials_from_file(_SERVICE_ACCOUNT_KEY_FILE)
    client = vision.ImageAnnotatorClient(credentials=credentials)
    with io.open(image_path, 'rb') as f:
        content = f.read()
    image = vision.Image(content=content)
    image_context = vision.ImageContext(language_hints=["en-t-i0-handwrit", "en"])
    response = client.document_text_detection(image=image, image_context=image_context)
    return response.full_text_annotation


# ─────────────────────────────────────────────────────────────
# WORD-LEVEL DATA EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_word_level_data(document_annotation) -> List[Dict]:
    word_data = []
    for page in document_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    text = "".join(s.text for s in word.symbols)
                    bbox = [(v.x, v.y) for v in word.bounding_box.vertices]
                    min_x = min(v[0] for v in bbox)
                    min_y = min(v[1] for v in bbox)
                    max_x = max(v[0] for v in bbox)
                    max_y = max(v[1] for v in bbox)
                    last_symbol = word.symbols[-1]
                    break_type = last_symbol.property.detected_break.type_
                    word_data.append({
                        'text': text,
                        'x': min_x,
                        'y': min_y,
                        'max_x': max_x,
                        'max_y': max_y,
                        'break_type': break_type,
                        'has_space_after': break_type in [
                            vision.TextAnnotation.DetectedBreak.BreakType.SPACE,
                            vision.TextAnnotation.DetectedBreak.BreakType.EOL_SURE_SPACE,
                            vision.TextAnnotation.DetectedBreak.BreakType.LINE_BREAK,
                        ]
                    })
    return word_data


# ─────────────────────────────────────────────────────────────
# LEGACY HELPERS  (kept for backward compatibility)
# ─────────────────────────────────────────────────────────────

# STRICT RULE: Q + number only. Students write Q1, Q2, Q3.
# OCR may read Q as q, O, or 0 — all accepted.
# Bare numbers, "1:", "1." etc. are NOT accepted — too error-prone.
_LABEL_PATTERN = re.compile(r'^[Qq0O](\d+)$')

def is_question_label(text: str) -> Optional[int]:
    """
    Return question number if text matches QN format, else None.
    Accepted: Q1 q1 Q12 O1 01  (common OCR variants of Q)
    Rejected: 1  1: 1.  any bare number or delimiter-only format
    """
    m = _LABEL_PATTERN.match(text.strip())
    if m:
        try:
            q = int(m.group(1))
            if 1 <= q <= 50:
                return q
        except ValueError:
            pass
    return None


def find_all_question_labels(word_data, left_margin_threshold=400, max_expected_question=20):
    found = []
    for i, word in enumerate(word_data):
        if word['x'] > left_margin_threshold:
            continue
        q = is_question_label(word['text'])
        if q is not None and q <= max_expected_question:
            found.append({
                'label': f'Q{q}',
                'q_number': q,
                'y_start': word['y'],
                'x_start': word['x'],
                'word_index': i,
                'raw_text': word['text'],
            })
    found.sort(key=lambda x: x['y_start'])
    return found


def validate_question_sequence(boundaries, strict=True, expected_questions=None):
    if not boundaries:
        return False, [], ["No questions detected!"], {}
    q_numbers = [b['q_number'] for b in boundaries]
    found_set = set(q_numbers)
    q_sorted = sorted(q_numbers)
    min_q, max_q = min(q_numbers), max(q_numbers)
    expected_set = set(expected_questions) if expected_questions else set(range(min_q, max_q + 1))
    missing = sorted(expected_set - found_set)
    warnings = []
    info = {
        'found_questions': q_sorted,
        'writing_order': q_numbers,
        'out_of_order': q_numbers != q_sorted,
        'has_duplicates': len(q_numbers) != len(found_set),
        'min_question': min_q,
        'max_question': max_q,
    }
    if 1 not in found_set:
        warnings.append(f"Q1 not found; starts from Q{min_q}")
    if missing:
        warnings.append(f"Missing: {', '.join(f'Q{m}' for m in missing)}")
    if info['has_duplicates']:
        dups = [n for n in found_set if q_numbers.count(n) > 1]
        warnings.append(f"Duplicates: {', '.join(f'Q{d}' for d in dups)}")
    if info['out_of_order']:
        warnings.append(f"Out of order: {' → '.join(f'Q{n}' for n in q_numbers[:6])}")
    is_valid = not missing and not info['has_duplicates'] and (1 in found_set or not strict)
    return is_valid, missing, warnings, info


def clean_answer_text(text: str, q_number: int) -> str:
    for p in [rf'^[Qq@]?\s*{q_number}\s*[:\.\)]?\s*', rf'^{q_number}\s*[:\.\)]?\s*']:
        text = re.sub(p, '', text, count=1, flags=re.IGNORECASE)
    return text.lstrip(' :.-_°)]}#@').strip()


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def detect_and_segment_image(image_path: str, debug: bool = True, config: Dict = None) -> Dict:
    """
    Primary entry point called by utils.py.

    Routing:
        1. Try geometry-based segmentation (sheet_geometry_segmentation.py)
           Uses the printed grid of the structured answer sheet for
           structurally-guaranteed column/row assignment — no heuristics.

        2. Fall back to heuristic segmentation (enhanced_vision_segmentation.py)
           if the printed lines cannot be detected (very light scans, photos,
           or any image that isn't the structured answer sheet).
    """
    if config is None:
        config = {}

    document_annotation = get_document_annotation(image_path)
    word_data = extract_word_level_data(document_annotation)

    # ── Primary: geometry-based ───────────────────────────────
    try:
        from paper_valuation.api.sheet_geometry_segmentation import segment_answers_geometry
        result = segment_answers_geometry(image_path, word_data, config=config, debug=debug)

        # If we got answers (or page is genuinely blank) — trust geometry path
        if result['answers'] or not word_data:
            return result

        logging.warning("Geometry segmentation returned no answers; falling back to heuristic")

    except Exception as geo_error:
        logging.warning(f"Geometry segmentation failed ({geo_error}); using heuristic fallback")

    # ── Fallback: heuristic ───────────────────────────────────
    return _segment_heuristic(document_annotation, word_data, debug=debug, config=config)


# ─────────────────────────────────────────────────────────────
# HEURISTIC FALLBACK  (original Y-gap logic, kept as safety net)
# ─────────────────────────────────────────────────────────────

def _segment_heuristic(document_annotation, word_data, debug=True, config=None):
    """Y-gap heuristic segmentation — fallback when geometry detection fails."""
    if config is None:
        config = {}

    from paper_valuation.api.enhanced_vision_segmentation import reconstruct_answer_text_adaptive

    LEFT_MARGIN_THRESHOLD = config.get('left_margin_threshold', 400)
    STRICT_VALIDATION = config.get('strict_validation', False)
    EXPECTED_QUESTIONS = config.get('expected_questions', None)
    MAX_EXPECTED_QUESTION = config.get('max_expected_question', 20)
    question_types = config.get('question_types', {})
    default_type = config.get('default_answer_type', 'short')
    is_handwritten = config.get('is_handwritten', True)

    if debug:
        logging.info("=" * 60)
        logging.info("HEURISTIC SEGMENTATION (fallback)")
        logging.info("=" * 60)

    boundaries = find_all_question_labels(word_data, LEFT_MARGIN_THRESHOLD, MAX_EXPECTED_QUESTION)
    is_valid, missing, warnings, val_info = validate_question_sequence(
        boundaries, strict=STRICT_VALIDATION, expected_questions=EXPECTED_QUESTIONS
    )

    answers_unsorted = {}

    if not boundaries and word_data:
        text = reconstruct_answer_text_adaptive(
            word_data, 0, len(word_data),
            is_handwritten=is_handwritten, answer_type='long'
        )
        if text.strip():
            answers_unsorted['UNLABELED_CONTINUATION'] = text

    for i, boundary in enumerate(boundaries):
        start_idx = boundary['word_index']
        if start_idx < len(word_data) and is_question_label(word_data[start_idx]['text']) is not None:
            start_idx += 1
        end_idx = boundaries[i + 1]['word_index'] if i + 1 < len(boundaries) else len(word_data)
        q_num = boundary['q_number']
        q_label = boundary['label']
        answer_type = question_types.get(str(q_num), default_type)
        text = reconstruct_answer_text_adaptive(
            word_data, start_idx, end_idx,
            is_handwritten=is_handwritten, answer_type=answer_type
        )
        text = clean_answer_text(text, q_num)

        # RULE: same Q label on the same page = continuation → concatenate
        if q_label in answers_unsorted:
            if text:
                answers_unsorted[q_label] = answers_unsorted[q_label] + ' ' + text
        else:
            answers_unsorted[q_label] = text

    sorted_keys = sorted(
        [k for k in answers_unsorted if k != 'UNLABELED_CONTINUATION'],
        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
    )
    answers = {}
    if 'UNLABELED_CONTINUATION' in answers_unsorted:
        answers['UNLABELED_CONTINUATION'] = answers_unsorted['UNLABELED_CONTINUATION']
    for k in sorted_keys:
        answers[k] = answers_unsorted[k]

    return {
        'answers': answers,
        'metadata': {
            'total_questions_found': len(boundaries),
            'question_numbers': val_info.get('found_questions', []),
            'writing_order': val_info.get('writing_order', []),
            'out_of_order': val_info.get('out_of_order', False),
            'is_complete': is_valid,
            'missing_questions': missing,
            'has_duplicates': val_info.get('has_duplicates', False),
            'is_handwritten': is_handwritten,
        },
        'validation': {'is_valid': is_valid, 'warnings': warnings, 'info': val_info}
    }


def segment_answers(document_annotation, debug=True, config=None):
    """Legacy alias — prefer detect_and_segment_image()."""
    word_data = extract_word_level_data(document_annotation)
    return _segment_heuristic(document_annotation, word_data, debug=debug, config=config)

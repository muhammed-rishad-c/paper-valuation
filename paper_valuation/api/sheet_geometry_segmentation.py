# sheet_geometry_segmentation.py
#
# Coordinate-space segmentation for the structured answer sheet.
#
# Sheet layout (measured from the provided A4 template):
#   - Vertical divider at ~10.8% of page width (left column = Q-label, right = answer)
#   - Horizontal lines create uniform rows (~25px each at 547px page width)
#   - 30–32 content rows per page
#
# Strategy:
#   1. Auto-detect divider X and row Y boundaries from the image itself
#      using morphological line detection — no hardcoded pixel values.
#   2. Assign every OCR word to a (row, column) cell using bounding-box centre.
#   3. Left-column words → question label candidates (structurally enforced).
#   4. Right-column words → answer content.
#   5. Group consecutive rows by which question label owns them.
#   6. Short answers: flat string join.
#      Long answers: row boundaries from the printed lines = paragraph separators
#                    (no Y-gap heuristics needed).

import re
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from paper_valuation.logging.logger import logging


# ─────────────────────────────────────────────────────────────
# SHEET GEOMETRY DETECTION
# ─────────────────────────────────────────────────────────────

class SheetGeometry:
    """
    Detected layout of the structured answer sheet.

    Attributes:
        divider_x   : pixel X of the vertical column divider
        row_ys      : sorted list of Y pixel positions of horizontal lines
        page_width  : image width in pixels
        page_height : image height in pixels
        divider_ratio: divider_x / page_width (scale-independent)
    """
    def __init__(self, divider_x: int, row_ys: List[int], page_width: int, page_height: int):
        self.divider_x = divider_x
        self.row_ys = row_ys
        self.page_width = page_width
        self.page_height = page_height
        self.divider_ratio = divider_x / page_width if page_width else 0.108

    def row_index_for_y(self, y: float) -> int:
        """Return which row band a Y coordinate falls in (0-indexed)."""
        for i in range(len(self.row_ys) - 1):
            if self.row_ys[i] <= y < self.row_ys[i + 1]:
                return i
        return len(self.row_ys) - 1

    def is_label_column(self, x: float) -> bool:
        """True if X is in the left (question-label) column."""
        return x < self.divider_x

    def __repr__(self):
        return (f"SheetGeometry(divider_x={self.divider_x}, "
                f"rows={len(self.row_ys)}, "
                f"page={self.page_width}x{self.page_height})")


def detect_sheet_geometry(image_path: str, divider_ratio_fallback: float = 0.108) -> SheetGeometry:
    """
    Auto-detect the sheet's column divider X and row Y boundaries
    from the actual image using morphological line detection.

    Falls back to ratio-based estimation if lines cannot be found
    (e.g., image is very light or over-compressed).

    Args:
        image_path            : path to the answer sheet image
        divider_ratio_fallback: fallback divider position as fraction of width

    Returns:
        SheetGeometry object with all layout coordinates
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold — handles varying scan brightness
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── Detect vertical divider ──────────────────────────────
    # Kernel height = 1/8 of page → only detects long vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 8, 20)))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    v_proj = np.sum(v_lines, axis=0)

    # Find inner vertical lines (exclude page borders within 5% of edges)
    margin = int(w * 0.05)
    inner_v = [(int(x), int(v_proj[x])) for x in range(margin, w - margin)
               if v_proj[x] > h * 0.25]

    divider_x: int
    if inner_v:
        # Take the leftmost strong vertical line as the label/answer divider
        divider_x = min(x for x, _ in inner_v)
        logging.info(f"  Sheet geometry: detected divider at X={divider_x} "
                     f"({divider_x/w:.3f} of width)")
    else:
        divider_x = int(w * divider_ratio_fallback)
        logging.warning(f"  Sheet geometry: divider not detected, "
                        f"using fallback X={divider_x}")

    # ── Detect horizontal row lines ──────────────────────────
    # Kernel width = 1/8 of page → only detects long horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 8, 20), 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    h_proj = np.sum(h_lines, axis=1)

    raw_ys = np.where(h_proj > w * 0.25)[0]

    # Cluster pixels that belong to the same printed line
    row_ys: List[int] = []
    if len(raw_ys) > 0:
        cluster_start = int(raw_ys[0])
        cluster_vals = [int(raw_ys[0])]
        for y in raw_ys[1:]:
            y = int(y)
            if y - cluster_vals[-1] <= 5:
                cluster_vals.append(y)
            else:
                row_ys.append(int(np.mean(cluster_vals)))
                cluster_start = y
                cluster_vals = [y]
        row_ys.append(int(np.mean(cluster_vals)))

    if len(row_ys) < 3:
        # Fallback: evenly-spaced rows
        logging.warning("  Sheet geometry: horizontal lines not detected, using fallback rows")
        estimated_row_height = 25  # px at ~547px width; scales with image
        row_ys = list(range(0, h, estimated_row_height))

    logging.info(f"  Sheet geometry: {len(row_ys)} row boundaries, "
                 f"avg row height {np.diff(row_ys).mean():.1f}px")

    return SheetGeometry(divider_x, row_ys, w, h)


# ─────────────────────────────────────────────────────────────
# WORD → CELL ASSIGNMENT
# ─────────────────────────────────────────────────────────────

class WordCell:
    """A word from OCR with its assigned sheet cell (row, column)."""
    __slots__ = ('text', 'x', 'y', 'max_x', 'max_y', 'row', 'is_label_col', 'has_space_after', 'break_type')

    def __init__(self, text, x, y, max_x, max_y, row, is_label_col, has_space_after, break_type):
        self.text = text
        self.x = x
        self.y = y
        self.max_x = max_x
        self.max_y = max_y
        self.row = row
        self.is_label_col = is_label_col
        self.has_space_after = has_space_after
        self.break_type = break_type


def assign_words_to_cells(word_data: List[Dict], geometry: SheetGeometry) -> List[WordCell]:
    """
    Map each OCR word to a (row, column) cell using its bounding-box centre.

    The centre point gives a more stable assignment than using top-left corner,
    especially for tall handwritten characters that cross line boundaries.
    """
    cells = []
    for word in word_data:
        centre_y = (word['y'] + word['max_y']) / 2
        centre_x = (word['x'] + word['max_x']) / 2

        row = geometry.row_index_for_y(centre_y)
        is_label = geometry.is_label_column(centre_x)

        cells.append(WordCell(
            text=word['text'],
            x=word['x'],
            y=word['y'],
            max_x=word['max_x'],
            max_y=word['max_y'],
            row=row,
            is_label_col=is_label,
            has_space_after=word.get('has_space_after', True),
            break_type=word.get('break_type', None),
        ))
    return cells


# ─────────────────────────────────────────────────────────────
# QUESTION LABEL DETECTION
# ─────────────────────────────────────────────────────────────

# STRICT RULE: Labels must be exactly "Q" + number (capital or lowercase Q).
# No other formats accepted — no bare numbers, no delimiters alone.
# Students are instructed to write Q1, Q2, Q3 etc.
# OCR commonly reads Q as q or occasionally O/0 — we handle those.
_LABEL_PATTERN = re.compile(r'^[Qq0O](\d+)$')

def parse_question_label(text: str) -> Optional[int]:
    """
    Return question number if text is a valid Q-label, else None.

    Accepted:  Q1  q1  Q12  (OCR variants: O1, 01)
    Rejected:  1   1:  1.   any bare number or other format
    """
    text = text.strip()
    m = _LABEL_PATTERN.match(text)
    if m:
        try:
            q = int(m.group(1))
            if 1 <= q <= 50:
                return q
        except ValueError:
            pass
    return None


def extract_row_labels(cells: List[WordCell], max_q: int = 50) -> Dict[int, int]:
    """
    Scan all left-column cells and return {row_index: question_number}.

    A question label is defined as any left-column word that parses as
    a valid Q-number. Multiple words in the same row that look like labels
    (rare OCR artifact) take the first match.
    """
    row_to_qnum: Dict[int, int] = {}
    for cell in cells:
        if not cell.is_label_col:
            continue
        q = parse_question_label(cell.text)
        if q is not None and q <= max_q:
            if cell.row not in row_to_qnum:
                row_to_qnum[cell.row] = q
    return row_to_qnum


# ─────────────────────────────────────────────────────────────
# QUESTION SPAN DETECTION
# ─────────────────────────────────────────────────────────────

def build_question_spans(row_to_qnum: Dict[int, int], total_rows: int) -> List[Dict]:
    """
    Given a mapping of {row: question_number}, build a list of spans:
        [{'q_number': N, 'start_row': R, 'end_row': E}, ...]

    A question spans from its label row until the row BEFORE the next label.
    The last question spans to the end of the page.

    Questions detected out of order are sorted by their row position,
    not their number — a student may write Q3 before Q2.
    """
    if not row_to_qnum:
        return []

    # Sort by row position
    sorted_labels = sorted(row_to_qnum.items(), key=lambda x: x[0])

    spans = []
    for i, (row, q_num) in enumerate(sorted_labels):
        if i + 1 < len(sorted_labels):
            next_row = sorted_labels[i + 1][0]
            end_row = next_row - 1
        else:
            end_row = total_rows - 1

        spans.append({
            'q_number': q_num,
            'label': f'Q{q_num}',
            'start_row': row,
            'end_row': end_row,
        })

    return spans


# ─────────────────────────────────────────────────────────────
# ANSWER TEXT RECONSTRUCTION
# ─────────────────────────────────────────────────────────────

def _words_for_span(cells: List[WordCell], start_row: int, end_row: int) -> List[WordCell]:
    """Return right-column cells within the given row range, sorted by position."""
    return sorted(
        [c for c in cells if not c.is_label_col and start_row <= c.row <= end_row],
        key=lambda c: (c.row, c.x)
    )


def reconstruct_short_answer(cells: List[WordCell], start_row: int, end_row: int) -> str:
    """
    Short answer: join all right-column words in the span into a flat string.
    Row boundaries are ignored — a short answer is one semantic unit.
    """
    words = _words_for_span(cells, start_row, end_row)
    if not words:
        return ''

    parts = []
    for i, cell in enumerate(words):
        parts.append(cell.text)
        if i < len(words) - 1:
            parts.append(' ')

    return re.sub(r' +', ' ', ''.join(parts)).strip()


def reconstruct_long_answer(cells: List[WordCell], start_row: int, end_row: int) -> str:
    """
    Long answer: use the PRINTED ROW BOUNDARIES as paragraph separators.

    Strategy:
    - Group right-column words by their row index.
    - Join words within each row as a single line.
    - Detect paragraph breaks between rows:
        * If the next row is empty (student left a blank line) → paragraph break.
        * If consecutive rows are filled → space-join (same paragraph).

    This completely replaces the Y-gap heuristic, using the sheet's own
    printed grid as ground truth.
    """
    words = _words_for_span(cells, start_row, end_row)
    if not words:
        return ''

    # Group words by row
    rows_map: Dict[int, List[WordCell]] = {}
    for cell in words:
        rows_map.setdefault(cell.row, []).append(cell)

    # Sort rows
    occupied_rows = sorted(rows_map.keys())
    if not occupied_rows:
        return ''

    # Build text with paragraph detection
    lines: List[str] = []
    for row in occupied_rows:
        row_words = sorted(rows_map[row], key=lambda c: c.x)
        line = ' '.join(c.text for c in row_words)
        line = re.sub(r' +', ' ', line).strip()
        if line:
            lines.append((row, line))

    if not lines:
        return ''

    parts = []
    for i, (row, line) in enumerate(lines):
        parts.append(line)
        if i < len(lines) - 1:
            next_row = lines[i + 1][0]
            gap = next_row - row  # number of sheet rows skipped

            if gap > 1:
                # Student left ≥1 blank row → paragraph break
                parts.append('\n\n')
            else:
                # Consecutive row → same paragraph, space-join
                parts.append(' ')

    text = ''.join(parts)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────
# MAIN SEGMENTATION ENTRY POINT
# ─────────────────────────────────────────────────────────────

def segment_answers_geometry(
    image_path: str,
    word_data: List[Dict],
    config: Dict = None,
    debug: bool = True
) -> Dict:
    """
    Geometry-aware segmentation for the structured answer sheet.

    Replaces the heuristic-based segment_answers() in vision_segmentation.py.
    Called from detect_and_segment_image() when the image path is available.

    Args:
        image_path : path to the answer sheet image (needed for geometry detection)
        word_data  : OCR word list from extract_word_level_data()
        config     : same config dict as before
                     (question_types, is_handwritten, max_expected_question, ...)
        debug      : print diagnostic info

    Returns:
        Same dict structure as segment_answers() for full drop-in compatibility:
        {
            'answers': {'Q1': '...', 'Q2': '...', ...},
            'metadata': {...},
            'validation': {...}
        }
    """
    if config is None:
        config = {}

    question_types = config.get('question_types', {})
    default_type = config.get('default_answer_type', 'short')
    is_handwritten = config.get('is_handwritten', True)
    max_q = config.get('max_expected_question', 50)

    # ── Step 1: Detect sheet geometry ────────────────────────
    geometry = detect_sheet_geometry(image_path)

    if debug:
        logging.info("=" * 70)
        logging.info(f"GEOMETRY-BASED SEGMENTATION")
        logging.info(f"  {geometry}")
        logging.info(f"  Document type: {'HANDWRITTEN' if is_handwritten else 'PRINTED'}")
        logging.info("=" * 70)

    # ── Step 2: Assign words to cells ────────────────────────
    cells = assign_words_to_cells(word_data, geometry)

    # ── Step 3: Find question labels in left column ──────────
    row_to_qnum = extract_row_labels(cells, max_q=max_q)

    if debug:
        logging.info(f"Question labels found: "
                     f"{[(f'Q{q}', f'row {r}') for r, q in sorted(row_to_qnum.items())]}")

    total_rows = len(geometry.row_ys)

    # ── Step 4: Build question row spans ─────────────────────
    spans = build_question_spans(row_to_qnum, total_rows)

    # ── Step 5: Handle pages with NO labels (continuation) ───
    if not spans:
        # Treat full page as continuation of previous answer (long-style)
        all_right = [c for c in cells if not c.is_label_col]
        all_right.sort(key=lambda c: (c.row, c.x))
        text = ' '.join(c.text for c in all_right)
        text = re.sub(r' +', ' ', text).strip()

        if debug:
            logging.info("⚠️  No question labels — treating page as UNLABELED_CONTINUATION")

        return {
            'answers': {'UNLABELED_CONTINUATION': text} if text else {},
            'metadata': {
                'total_questions_found': 0,
                'question_numbers': [],
                'writing_order': [],
                'out_of_order': False,
                'is_complete': False,
                'missing_questions': [],
                'has_duplicates': False,
                'is_handwritten': is_handwritten,
                'geometry': str(geometry),
            },
            'validation': {
                'is_valid': False,
                'warnings': ['No question labels found on this page'],
                'info': {}
            }
        }

    # ── Step 6: Reconstruct each answer ──────────────────────
    # RULE: If the same Q label appears multiple times on this page
    # (student continued on next section), concatenate the text.
    answers: Dict[str, str] = {}
    found_q_numbers = []

    for span in spans:
        q_num = span['q_number']
        q_label = span['label']
        answer_type = question_types.get(str(q_num), default_type)

        if answer_type == 'long':
            text = reconstruct_long_answer(cells, span['start_row'], span['end_row'])
        else:
            text = reconstruct_short_answer(cells, span['start_row'], span['end_row'])

        # Clean any accidental label prefix that OCR put in the answer area
        text = re.sub(rf'^[Qq0O]?\s*{q_num}\s*[:\.\)]?\s*', '', text, count=1).strip()

        if q_label in answers:
            # Same Q label seen again on this page — append (continuation)
            if text:
                answers[q_label] = answers[q_label] + ' ' + text
            if debug:
                logging.info(f"  {q_label} continuation on same page — concatenated")
        else:
            answers[q_label] = text
            found_q_numbers.append(q_num)

        if debug:
            preview = text[:120] + ('...' if len(text) > 120 else '')
            logging.info(f"  {q_label} [{answer_type}] rows {span['start_row']}–{span['end_row']}: {preview}")

    # Sort by question number
    answers = dict(sorted(answers.items(),
                          key=lambda kv: int(re.search(r'\d+', kv[0]).group())))

    # ── Step 7: Validation ────────────────────────────────────
    # Note: duplicate Q labels on the same page are VALID (continuation) — not errors
    writing_order = [s['q_number'] for s in spans]
    out_of_order = sorted(writing_order) != writing_order

    warnings = []
    if out_of_order:
        warnings.append(f"Non-sequential order: {writing_order}")

    if debug:
        logging.info(f"\nSUMMARY: {len(answers)} answers extracted | "
                     f"{'✅ ordered' if not out_of_order else '⚠️ out of order'}")

    return {
        'answers': answers,
        'metadata': {
            'total_questions_found': len(found_q_numbers),
            'question_numbers': sorted(found_q_numbers),
            'writing_order': writing_order,
            'out_of_order': out_of_order,
            'is_complete': True,
            'missing_questions': [],
            'has_duplicates': False,   # same-Q continuation is valid, not a duplicate error
            'is_handwritten': is_handwritten,
            'geometry': str(geometry),
        },
        'validation': {
            'is_valid': True,
            'warnings': warnings,
            'info': {
                'found_questions': sorted(found_q_numbers),
                'writing_order': writing_order,
                'out_of_order': out_of_order,
                'has_duplicates': False,
            }
        }
    }

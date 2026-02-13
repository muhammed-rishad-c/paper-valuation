# enhanced_vision_segmentation.py
# IMPROVED VERSION with Smart Adaptive Paragraph Detection

from google.cloud import vision
import re
from typing import Dict, List

# Expanded bullet markers list
BULLET_MARKERS = ['•', '●', '○', '-', '*', '→', '▸', '>', '■', '□', '▪', '◆', '◇', '►', '»', '–', '—']

def calculate_average_line_height(words: List[Dict]) -> float:
    """
    Calculate average line height from word bounding boxes
    Used for adaptive paragraph detection
    """
    line_heights = []
    
    for word in words:
        height = word['max_y'] - word['y']
        if height > 5:  # Filter out noise
            line_heights.append(height)
    
    return sum(line_heights) / len(line_heights) if line_heights else 40


def calculate_dominant_x_position(words: List[Dict]) -> float:
    """
    Find the most common X position (left margin)
    Helps detect indentation
    """
    x_positions = [word['x'] for word in words]
    
    # Group X positions within 10px tolerance
    x_groups = {}
    for x in x_positions:
        found_group = False
        for key in x_groups:
            if abs(x - key) < 10:
                x_groups[key].append(x)
                found_group = True
                break
        if not found_group:
            x_groups[x] = [x]
    
    # Find largest group
    if not x_groups:
        return 0
    
    largest_group = max(x_groups.values(), key=len)
    return sum(largest_group) / len(largest_group)


def detect_paragraph_signals(current_word: Dict, next_word: Dict, 
                            avg_line_height: float, dominant_x: float) -> Dict:
    """
    Multi-signal paragraph boundary detection
    
    Returns dictionary with individual signals and combined score
    """
    y_gap = next_word['y'] - current_word['max_y']
    x_shift = abs(next_word['x'] - dominant_x)
    
    # Individual signals
    signals = {
        'large_gap': y_gap > (avg_line_height * 1.3),
        'moderate_gap': y_gap > (avg_line_height * 0.8),
        'is_bullet': next_word['text'].strip() in BULLET_MARKERS or \
                     (len(next_word['text']) > 0 and next_word['text'][0] in BULLET_MARKERS),
        'indented': x_shift > 25,
        'outdented': next_word['x'] < (current_word['x'] - 15),
        'sentence_end': current_word['text'].rstrip().endswith(('.', '!', '?', ':')),
        'line_break': current_word['break_type'] in [
            vision.TextAnnotation.DetectedBreak.BreakType.LINE_BREAK,
            vision.TextAnnotation.DetectedBreak.BreakType.EOL_SURE_SPACE
        ],
        'current_line_height': current_word['max_y'] - current_word['y'],
        'y_gap_pixels': y_gap
    }
    
    # Scoring system (0-5 points)
    score = 0
    
    # Strong signals (2 points each)
    if signals['large_gap']:
        score += 2
    if signals['is_bullet']:
        score += 2
    
    # Moderate signals (1 point each)
    if signals['indented'] and signals['line_break']:
        score += 1
    if signals['sentence_end'] and signals['moderate_gap']:
        score += 1
    if signals['outdented'] and signals['moderate_gap']:
        score += 1
    
    # Special case: Very large gap (extra point)
    if y_gap > (avg_line_height * 2):
        score += 1
    
    signals['paragraph_break_score'] = score
    signals['should_break'] = score >= 2  # Threshold: 2+ points = paragraph break
    
    return signals


def reconstruct_answer_text_adaptive(words: List[Dict], start_idx: int, 
                                    end_idx: int = None, 
                                    is_handwritten: bool = True) -> str:
    """
    ENHANCED VERSION: Adaptive paragraph reconstruction
    
    Improvements over original:
    1. Calculates average line height from actual document
    2. Multi-signal paragraph detection
    3. Better bullet point handling
    4. Indentation awareness
    5. Adaptive spacing thresholds
    """
    if end_idx is None:
        end_idx = len(words)
    
    # Learn from document
    answer_words = words[start_idx:end_idx]
    avg_line_height = calculate_average_line_height(answer_words)
    dominant_x = calculate_dominant_x_position(answer_words)
    
    print(f"   📏 Adaptive Detection:")
    print(f"      Avg line height: {avg_line_height:.1f}px")
    print(f"      Dominant X position: {dominant_x:.1f}px")
    print(f"      Document type: {'Handwritten' if is_handwritten else 'Printed'}")
    
    answer_parts = []
    paragraph_breaks_detected = 0
    
    for i in range(start_idx, min(end_idx, len(words))):
        word = words[i]
        answer_parts.append(word['text'])
        
        if i < end_idx - 1:
            next_word = words[i + 1]
            
            # Multi-signal detection
            signals = detect_paragraph_signals(word, next_word, avg_line_height, dominant_x)
            
            if signals['should_break']:
                answer_parts.append('\n\n')
                paragraph_breaks_detected += 1
                
                # Debug logging
                if signals['is_bullet']:
                    print(f"      ✓ Paragraph break before bullet '{next_word['text']}'")
                elif signals['large_gap']:
                    print(f"      ✓ Paragraph break: Large gap ({signals['y_gap_pixels']:.0f}px)")
                
            elif signals['line_break']:
                # Just a line break, not a paragraph
                answer_parts.append(' ')
            else:
                # Same line
                if word['has_space_after']:
                    answer_parts.append(' ')
    
    # Cleanup
    text = ''.join(answer_parts)
    text = re.sub(r' +', ' ', text)  # Multiple spaces → single space
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines → double newline
    text = re.sub(r' +\n', '\n', text)  # Trailing spaces
    text = re.sub(r'\n +', '\n', text)  # Leading spaces
    
    print(f"      📊 Detected {paragraph_breaks_detected} paragraph breaks")
    
    return text.strip()


def validate_answer_format(document_annotation, strict_mode: bool = False, 
                          expected_questions: List[int] = None) -> tuple:
    """
    Validate answer sheet formatting
    
    Returns:
        (is_valid, errors, warnings)
    """
    from paper_valuation.api.vision_segmentation import (
        extract_word_level_data,
        find_all_question_labels
    )
    
    word_data = extract_word_level_data(document_annotation)
    boundaries = find_all_question_labels(word_data)
    
    errors = []
    warnings = []
    
    if not boundaries:
        errors.append("No question labels found! Please write Q1, Q2, etc. on the left side.")
        return False, errors, warnings
    
    # Check question labels
    question_numbers = [b['q_number'] for b in boundaries]
    
    # Check for duplicates
    if len(question_numbers) != len(set(question_numbers)):
        duplicates = [q for q in set(question_numbers) if question_numbers.count(q) > 1]
        errors.append(f"Duplicate question numbers found: {', '.join([f'Q{d}' for d in duplicates])}")
    
    # Check sequence
    if question_numbers != sorted(question_numbers):
        warnings.append(f"Questions out of order: {' → '.join([f'Q{q}' for q in question_numbers])}")
    
    # Check spacing between questions
    for i in range(len(boundaries) - 1):
        spacing = boundaries[i + 1]['y_start'] - boundaries[i]['y_start']
        
        if spacing < 50:
            warnings.append(f"Q{boundaries[i]['q_number']} and Q{boundaries[i+1]['q_number']} are too close. Leave more space.")
        
        if strict_mode and spacing < 100:
            errors.append(f"Strict mode: Q{boundaries[i]['q_number']} and Q{boundaries[i+1]['q_number']} must have at least 3 lines of spacing")
    
    # Check for proper bullet formatting in long answers (if we know which are long)
    # This would require answer key metadata
    
    is_valid = len(errors) == 0
    
    return is_valid, errors, warnings


def generate_formatting_suggestions(errors: List[str], warnings: List[str]) -> List[str]:
    """
    Generate helpful suggestions based on validation results
    """
    suggestions = []
    
    if any("No question labels" in e for e in errors):
        suggestions.append("Write 'Q1:', 'Q2:', etc. at the START of each answer on the LEFT side")
    
    if any("Duplicate" in e for e in errors):
        suggestions.append("Check your question numbers - each question should appear only once")
    
    if any("too close" in w for w in warnings):
        suggestions.append("Leave MORE SPACE between questions (at least 2-3 blank lines)")
    
    if any("out of order" in w for w in warnings):
        suggestions.append("Write questions in order: Q1, Q2, Q3... (or clearly mark continuation pages)")
    
    if not suggestions:
        suggestions.append("Format looks good! Remember to use bullet points for long answers.")
    
    return suggestions


def normalize_bullet_points(text: str) -> str:
    """
    Convert various bullet markers to standard format
    Helps with consistent display
    """
    for bullet in BULLET_MARKERS:
        if bullet != '•':  # Keep • as standard
            text = text.replace(bullet, '•')
    
    return text


def count_paragraph_breaks(text: str) -> int:
    """
    Count double-newline paragraph breaks
    """
    return text.count('\n\n')


def analyze_answer_structure(text: str) -> Dict:
    """
    Analyze the structure of an answer
    Returns metadata about formatting
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    bullet_count = sum(1 for p in paragraphs if any(p.startswith(b) for b in BULLET_MARKERS))
    
    return {
        'paragraph_count': len(paragraphs),
        'bullet_point_count': bullet_count,
        'avg_paragraph_length': sum(len(p) for p in paragraphs) / len(paragraphs) if paragraphs else 0,
        'total_length': len(text),
        'has_bullets': bullet_count > 0,
        'is_structured': bullet_count >= 2  # Multiple bullet points = well-structured
    }


# ===============================================
# INTEGRATION HELPER
# ===============================================

def segment_answers_enhanced(document_annotation, debug=True, config=None):
    """
    Enhanced wrapper that uses adaptive reconstruction
    
    This is a DROP-IN REPLACEMENT for the original segment_answers()
    Just change the import in your code:
    
    OLD: from vision_segmentation import segment_answers
    NEW: from enhanced_vision_segmentation import segment_answers_enhanced as segment_answers
    """
    from paper_valuation.api.vision_segmentation import (
        extract_word_level_data,
        find_all_question_labels,
        validate_question_sequence,
        is_question_label,
        clean_answer_text
    )
    
    if config is None:
        config = {}
    
    # Get configuration
    is_handwritten = config.get('is_handwritten', True)
    question_types = config.get('question_types', {})
    default_answer_type = config.get('default_answer_type', 'short')
    
    # Extract words
    word_data = extract_word_level_data(document_annotation)
    
    if debug:
        print(f"\n{'='*70}")
        print(f"📄 ENHANCED ADAPTIVE SEGMENTATION")
        print(f"   Words extracted: {len(word_data)}")
        print(f"   Document type: {'Handwritten' if is_handwritten else 'Printed'}")
        print(f"{'='*70}")
    
    # Find question labels
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
    
    # Validate
    is_valid, missing, warnings, validation_info = validate_question_sequence(
        boundaries,
        strict=config.get('strict_validation', False),
        expected_questions=config.get('expected_questions', None)
    )
    
    if debug:
        print(f"\n✓ VALIDATION:")
        for warning in warnings:
            print(f"   {warning}")
    
    # Segment answers using ADAPTIVE reconstruction
    segmented_answers = {}
    
    for i, boundary in enumerate(boundaries):
        start_idx = boundary['word_index']
        
        # Skip the question label itself
        if start_idx < len(word_data) and is_question_label(word_data[start_idx]['text']) is not None:
            start_idx += 1
        
        # End index
        end_idx = boundaries[i + 1]['word_index'] if i + 1 < len(boundaries) else len(word_data)
        
        # Get question metadata
        q_num = boundary['q_number']
        answer_type = question_types.get(str(q_num), default_answer_type)
        
        if debug:
            print(f"\n   Processing {boundary['label']} ({answer_type} answer)...")
        
        # Use ADAPTIVE reconstruction
        answer_text = reconstruct_answer_text_adaptive(
            word_data, 
            start_idx, 
            end_idx,
            is_handwritten=is_handwritten
        )
        
        # Clean
        answer_text = clean_answer_text(answer_text, q_num)
        
        # Analyze structure
        structure = analyze_answer_structure(answer_text)
        
        if debug:
            print(f"      Structure: {structure['paragraph_count']} paragraphs, "
                  f"{structure['bullet_point_count']} bullets")
        
        segmented_answers[boundary['label']] = answer_text
    
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



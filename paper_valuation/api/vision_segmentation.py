from google.cloud import vision
import google.auth
import io
import os
import re
from typing import Dict, List, Optional, Tuple
from paper_valuation.api.enhanced_vision_segmentation import reconstruct_answer_text_adaptive
from paper_valuation.logging.logger import logging

from dotenv import load_dotenv
load_dotenv()

_SERVICE_ACCOUNT_KEY_FILE = os.environ.get("SERVICE_ACCOUNT_KEY_FILE")

if not _SERVICE_ACCOUNT_KEY_FILE:
    raise ValueError("SERVICE_ACCOUNT_KEY_FILE not found. Check if .env is loaded correctly.")



def get_document_annotation(image_path: str):
    credentials, project_id = google.auth.load_credentials_from_file(_SERVICE_ACCOUNT_KEY_FILE)
    client = vision.ImageAnnotatorClient(credentials=credentials)
    
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
        
    image = vision.Image(content=content)
    image_context = vision.ImageContext(language_hints=["en-t-i0-handwrit", "en"])
    
    response = client.document_text_detection(
        image=image,
        image_context=image_context
    )
    return response.full_text_annotation



def is_question_label(text: str) -> Optional[int]:
    text = text.strip()
    
    patterns = [
        r'^[Qq@]?\s*(\d+)\s*[:\.\)]\s*$',  
        r'^[Qq@]?\s*(\d+)\s*[:\.\)]',       
        r'^[Qq@](\d+)$',                    
        r'^(\d+)$',
    ]
    
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            try:
                q_num = int(match.group(1))
                if 1 <= q_num <= 50: 
                    return q_num
            except ValueError:
                continue
    
    return None



def extract_word_level_data(document_annotation) -> List[Dict]:
    
    word_data = []
    
    for page in document_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    text = "".join([symbol.text for symbol in word.symbols])
                    
                    bbox = [(v.x, v.y) for v in word.bounding_box.vertices]
                    min_x = min(v[0] for v in bbox)
                    min_y = min(v[1] for v in bbox)
                    max_x = max(v[0] for v in bbox)
                    max_y = max(v[1] for v in bbox)
                    
                    last_symbol = word.symbols[-1]
                    break_type = last_symbol.property.detected_break.type_
                    
                    # Store full break type info for formatted reconstruction
                    word_data.append({
                        'text': text,
                        'x': min_x,
                        'y': min_y,
                        'max_x': max_x,
                        'max_y': max_y,
                        'break_type': break_type,  # Full break type (NEW)
                        'has_space_after': break_type in [
                            vision.TextAnnotation.DetectedBreak.BreakType.SPACE,
                            vision.TextAnnotation.DetectedBreak.BreakType.EOL_SURE_SPACE,
                            vision.TextAnnotation.DetectedBreak.BreakType.LINE_BREAK
                        ]
                    })
    return word_data



def find_all_question_labels(word_data: List[Dict], left_margin_threshold: int = 400, max_expected_question: int = 20) -> List[Dict]:

    found_labels = []
    
    for i, word in enumerate(word_data):
        if word['x'] > left_margin_threshold:
            continue
        
        q_num = is_question_label(word['text'])
        
        if q_num is not None:
            if q_num > max_expected_question:
                print(f"   ⚠️  Ignoring suspicious label: Q{q_num} (likely OCR error)")
                continue
                
            found_labels.append({
                'label': f'Q{q_num}',
                'q_number': q_num,
                'y_start': word['y'],
                'x_start': word['x'],
                'word_index': i,
                'raw_text': word['text']
            })
    
    found_labels.sort(key=lambda x: x['y_start'])
    
    return found_labels



def validate_question_sequence(boundaries: List[Dict], strict: bool = True, expected_questions: List[int] = None) -> Tuple[bool, List[int], List[str], Dict]:
    
    if not boundaries:
        return False, [], ["❌ No questions detected in the document!"], {}
    
    q_numbers = [b['q_number'] for b in boundaries]
    found_set = set(q_numbers)
    q_numbers_sorted = sorted(q_numbers)
    
    min_q = min(q_numbers)
    max_q = max(q_numbers)
    
    if expected_questions is not None:
        expected_set = set(expected_questions)
    else:
        expected_set = set(range(min_q, max_q + 1))
    
    missing = sorted(expected_set - found_set)
    extra = sorted(found_set - expected_set) if expected_questions else []
    
    warnings = []
    info = {
        'found_questions': q_numbers_sorted,
        'writing_order': q_numbers, 
        'out_of_order': q_numbers != q_numbers_sorted,
        'has_duplicates': len(q_numbers) != len(found_set),
        'min_question': min_q,
        'max_question': max_q
    }
    if 1 not in found_set:
        if strict:
            warnings.append(f"⚠️  Q1 not found. Document starts from Q{min_q}.")
        else:
            warnings.append(f"ℹ️  Document starts from Q{min_q} (continuation sheet)")
    
    if missing:
        warnings.append(f"❌ Missing questions: {', '.join([f'Q{m}' for m in missing])}")
    
    if extra and expected_questions is not None:
        warnings.append(f"ℹ️  Found unexpected questions: {', '.join([f'Q{e}' for e in extra])}")
    
    if info['has_duplicates']:
        duplicates = [n for n in found_set if q_numbers.count(n) > 1]
        warnings.append(f"⚠️  Duplicate labels detected: {', '.join([f'Q{d}' for d in duplicates])}")
    
    if info['out_of_order']:
        warnings.append(f"ℹ️  Questions written in non-sequential order: {' → '.join([f'Q{n}' for n in q_numbers[:5]])}{'...' if len(q_numbers) > 5 else ''}")
    
    for i in range(len(boundaries) - 1):
        spacing = boundaries[i + 1]['y_start'] - boundaries[i]['y_start']
        if spacing < 30:  # Very close together
            q1_num = boundaries[i]['q_number']
            q2_num = boundaries[i + 1]['q_number']
            warnings.append(f"⚠️  Q{q1_num} and Q{q2_num} are very close (spacing: {spacing}px). Add more space.")
    
    is_valid = len(missing) == 0 and not info['has_duplicates'] and (1 in found_set or not strict)
    
    return is_valid, missing, warnings, info


def reconstruct_answer_text_formatted(words: List[Dict], start_idx: int, end_idx: Optional[int] = None, 
                                      is_handwritten: bool = True) -> str:
    if end_idx is None:
        end_idx = len(words)
    
    answer_parts = []
    consecutive_newlines = 0
    
    PARAGRAPH_GAP_THRESHOLD = 45 if is_handwritten else 70  
    
    for i in range(start_idx, min(end_idx, len(words))):
        word = words[i]
        answer_parts.append(word['text'])
        
        if i < end_idx - 1:
            break_type = word['break_type']
            
            if break_type in [
                vision.TextAnnotation.DetectedBreak.BreakType.LINE_BREAK,
                vision.TextAnnotation.DetectedBreak.BreakType.EOL_SURE_SPACE
            ]:
                consecutive_newlines += 1
            
            elif break_type == vision.TextAnnotation.DetectedBreak.BreakType.SPACE:
                answer_parts.append(' ')
                consecutive_newlines = 0
            
            else:
                consecutive_newlines = 0

            if consecutive_newlines > 0:
                next_word = words[i + 1] if i + 1 < end_idx else None
                
                if next_word:
                    is_bullet = next_word['text'].strip() in ['•', '●', '○', '-', '*', '→', '▸'] or \
                                (len(next_word['text']) >= 2 and next_word['text'][0] in ['•', '●', '○', '-', '*'])
                    y_gap = next_word['y'] - word['max_y']
                    
                    current_line_height = word['max_y'] - word['y']
                    should_break = False
                    if is_bullet:
                        should_break = True
                    
                    elif y_gap > PARAGRAPH_GAP_THRESHOLD:
                        should_break = True
                    
                    elif current_line_height > 0 and y_gap > (current_line_height * 1.3):
                        should_break = True
                    
                    elif is_handwritten and abs(next_word['x'] - word['x']) > 50:
                        should_break = True

                    elif word['text'].rstrip().endswith(('.', '!', '?')) and y_gap > 25:
                        should_break = True
                    
                    if should_break:
                        answer_parts.append('\n\n')  
                    else:
                        answer_parts.append(' ')  
                    
                    consecutive_newlines = 0
    
    text = ''.join(answer_parts)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +\n', '\n', text)
    text = re.sub(r'\n +', '\n', text)
    
    return text.strip()



def clean_answer_text(text: str, q_number: int) -> str:
    patterns = [
        rf'^[Qq@]?\s*{q_number}\s*[:\.\)]?\s*',
        rf'^{q_number}\s*[:\.\)]?\s*',
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, count=1, flags=re.IGNORECASE)
    
    text = text.lstrip(' :.-_°)]}#@')
    return text.strip()


 
def segment_answers(document_annotation, debug: bool = True, config: Dict = None) -> Dict:
    if config is None:
        config = {}
    
    LEFT_MARGIN_THRESHOLD = config.get('left_margin_threshold', 400)
    STRICT_VALIDATION = config.get('strict_validation', False)
    EXPECTED_QUESTIONS = config.get('expected_questions', None)
    MAX_EXPECTED_QUESTION = config.get('max_expected_question', 20)
    
    question_types = config.get('question_types', {})
    default_answer_type = config.get('default_answer_type', 'short')
    
    is_handwritten = config.get('is_handwritten', True)  
    
    word_data = extract_word_level_data(document_annotation)
    
    if debug:
        logging.info("="*70)
        logging.info(f"DOCUMENT ANALYSIS - {len(word_data)} words")
        logging.info(f"Type: {'HANDWRITTEN' if is_handwritten else 'PRINTED'}")
        logging.info("="*70)
        if question_types:
            print(f"📋 Using answer key configuration:")
            short_count = len([t for t in question_types.values() if t == 'short'])
            long_count = len([t for t in question_types.values() if t == 'long'])
            print(f"   {short_count} short questions, {long_count} long questions")
        else:
            print(f"📋 Using default answer type: '{default_answer_type}' for all questions")
        print(f"{'='*70}")
    
    boundaries = find_all_question_labels(word_data, LEFT_MARGIN_THRESHOLD, MAX_EXPECTED_QUESTION)
    
    if debug:
        print(f"\n🔍 QUESTION DETECTION:")
        if boundaries:
            print(f"   Found {len(boundaries)} question label(s):")
            for b in boundaries:
                if question_types:
                    q_type = question_types.get(str(b['q_number']), 'default')
                else:
                    q_type = default_answer_type
                print(f"      • {b['label']} at Y={b['y_start']}, X={b['x_start']} [Type: {q_type}]")
        else:
            print("   ❌ No question labels detected on this page!")
    
    is_valid, missing, warnings, validation_info = validate_question_sequence(
        boundaries, 
        strict=STRICT_VALIDATION,
        expected_questions=EXPECTED_QUESTIONS
    )
    
    if debug:
        print(f"\n✓ VALIDATION:")
        found_q = validation_info.get('found_questions', [])
        if found_q:
            print(f"   Questions Found: {', '.join([f'Q{q}' for q in found_q])}")
        
        if validation_info.get('out_of_order'):
            writing_order = validation_info.get('writing_order', [])
            print(f"   Writing Order: {' → '.join([f'Q{q}' for q in writing_order[:5]])}")
        
        if is_valid:
            print(f"   ✅ All expected questions present on this page")
        
        for warning in warnings:
            print(f"   {warning}")
    
    segmented_answers_unsorted = {}
   
    if not boundaries and word_data:
        full_text = reconstruct_answer_text_adaptive(word_data, 0, len(word_data), is_handwritten=is_handwritten)  # ✅ CORRECT!
        if full_text.strip():
            segmented_answers_unsorted['UNLABELED_CONTINUATION'] = full_text
            if debug:
                print(f"\n⚠️  UNLABELED PAGE DETECTED:")
                preview = full_text[:100] + ('...' if len(full_text) > 100 else '')
                print(f"   This page has no question labels. Text: {preview}")
                print(f"   ⚠️  Student should write the question number at the top!")

    for i, boundary in enumerate(boundaries):
        start_idx = boundary['word_index']
        
        if start_idx < len(word_data):
            if is_question_label(word_data[start_idx]['text']) is not None:
                start_idx += 1
        
        if i + 1 < len(boundaries):
            end_idx = boundaries[i + 1]['word_index']
        else:
            end_idx = len(word_data)
        
        q_num = boundary['q_number']
        
        if question_types:
            answer_type = question_types.get(str(q_num), 'short')
        else:
            answer_type = default_answer_type
        
        if debug:
            print(f"\n   Processing {boundary['label']} as '{answer_type}' answer")
        
        # Use enhanced adaptive reconstruction for ALL answer types
        answer_text = reconstruct_answer_text_adaptive(
            word_data, start_idx, end_idx, is_handwritten=is_handwritten
        )
        
        answer_text = clean_answer_text(answer_text, boundary['q_number'])
        
        segmented_answers_unsorted[boundary['label']] = answer_text
    
    sorted_keys = sorted(
        [k for k in segmented_answers_unsorted.keys() if k != 'UNLABELED_CONTINUATION'],
        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
    )
    
    segmented_answers = {}
    
    if 'UNLABELED_CONTINUATION' in segmented_answers_unsorted:
        segmented_answers['UNLABELED_CONTINUATION'] = segmented_answers_unsorted['UNLABELED_CONTINUATION']
        
    for k in sorted_keys:
        segmented_answers[k] = segmented_answers_unsorted[k]
    
    if debug:
        print(f"\n📄 EXTRACTED ANSWERS:")
        for q_label, answer_text in segmented_answers.items():
            preview = answer_text[:150] + ('...' if len(answer_text) > 150 else '')
            print(f"   {q_label}: {preview}")
    
    result = {
        'answers': segmented_answers,
        'metadata': {
            'total_questions_found': len(boundaries),
            'question_numbers': validation_info.get('found_questions', []),
            'writing_order': validation_info.get('writing_order', []),
            'out_of_order': validation_info.get('out_of_order', False),
            'is_complete': is_valid,
            'missing_questions': missing,
            'has_duplicates': validation_info.get('has_duplicates', False),
            'expected_questions': EXPECTED_QUESTIONS,
            'is_handwritten': is_handwritten  # Include in metadata
        },
        'validation': {
            'is_valid': is_valid,
            'warnings': warnings,
            'info': validation_info
        }
    }
    if debug:
        print(f"\n{'='*70}")
        print(f"📊 SUMMARY: Found {len(segmented_answers)} answer parts")
        print(f"{'='*70}\n")
    
    return result



def detect_and_segment_image(image_path: str, debug: bool = True, config: Dict = None) -> Dict:

    document_annotation = get_document_annotation(image_path)
    result = segment_answers(document_annotation, debug=debug, config=config)
    return result
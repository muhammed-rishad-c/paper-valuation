import re
import json
import uuid
import os
import tempfile
from flask import jsonify, request
from paper_valuation.logging.logger import logging
from paper_valuation.api.vision_segmentation import detect_and_segment_image, get_document_annotation
from paper_valuation.components.valuation import (
    evaluation_short_answer,
    evaluation_long_answer,
    smart_paragraph_split
)

ANSWER_KEYS_FILE = 'answer_keys.json'

# ============================================
# ANSWER KEY MANAGEMENT
# ============================================

def load_answer_keys():
    if os.path.exists(ANSWER_KEYS_FILE):
        with open(ANSWER_KEYS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_answer_keys(answer_keys):
    with open(ANSWER_KEYS_FILE, 'w') as f:
        json.dump(answer_keys, f, indent=2)

def get_answer_key_by_id(exam_id):
    all_keys = load_answer_keys()
    return all_keys.get(exam_id, None)

def generate_exam_id(exam_name, class_name, subject):
    base = f"{subject}_{class_name}_{exam_name.replace(' ', '_')}"
    unique_suffix = str(uuid.uuid4())[:8].upper()
    return f"{base}_{unique_suffix}"

# ============================================
# PARSING UTILITIES
# ============================================

def parse_question_range(range_str):
    questions = []
    parts = range_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            questions.extend(range(start, end + 1))
        else:
            questions.append(int(part))
    return questions

def parse_marks_string(marks_str, question_count):
    if not marks_str or not marks_str.strip():
        raise ValueError("Marks string cannot be empty")
    
    marks_str = marks_str.strip()
    
    if ',' not in marks_str:
        try:
            mark = int(marks_str)
            if mark <= 0:
                raise ValueError("Marks must be positive numbers")
            return [mark] * question_count
        except ValueError:
            raise ValueError(f"Invalid marks format: '{marks_str}'.")
    
    marks_parts = [part.strip() for part in marks_str.split(',')]
    
    try:
        marks_list = [int(mark) for mark in marks_parts]
    except ValueError:
        raise ValueError(f"Invalid marks format: '{marks_str}'. All values must be numbers.")
    
    if any(mark <= 0 for mark in marks_list):
        raise ValueError("All marks must be positive numbers")
    
    if len(marks_list) != question_count:
        raise ValueError(
            f"Marks count mismatch: provided {len(marks_list)} marks but have {question_count} questions."
        )
    
    return marks_list

def normalize_question_key(key, add_prefix=True):
    if isinstance(key, int):
        num = str(key)
    else:
        match = re.search(r'\d+', str(key))
        num = match.group() if match else "0"
    return f"Q{num}" if add_prefix else num

# ============================================
# MULTI-PAGE MERGING
# ============================================

def merge_multi_page_result(all_pages_list):
    merged_answers = {}
    last_q_label = None

    for page_index, page in enumerate(all_pages_list):
        answers = page.get('answers', {})

        for q_label, text in answers.items():
            text = text.strip()
            if not text:
                continue

            if q_label == 'UNLABELED_CONTINUATION':
                if last_q_label and last_q_label in merged_answers:
                    merged_answers[last_q_label] += ' ' + text
                else:
                    merged_answers.setdefault('Q1', '')
                    merged_answers['Q1'] += (' ' if merged_answers['Q1'] else '') + text
                    last_q_label = 'Q1'
            elif q_label in merged_answers:
                merged_answers[q_label] += ' ' + text
                last_q_label = q_label
            else:
                merged_answers[q_label] = text
                last_q_label = q_label

    sorted_answers = dict(sorted(
        merged_answers.items(),
        key=lambda x: int(re.search(r'\d+', x[0]).group()) if re.search(r'\d+', x[0]) else 0
    ))

    return {"answers": sorted_answers, "total_pages": len(all_pages_list)}

# ============================================
# INDIVIDUAL EVALUATION
# ============================================

def evaluate_paper_individual(files, config=None):
    try:
        if config is None:
            config = {}
        if 'is_handwritten' not in config:
            config['is_handwritten'] = True

        all_page_result = []

        for index, file in enumerate(files):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                file.save(tmp.name)
                temp_path = tmp.name

            page_result = detect_and_segment_image(temp_path, debug=True, config=config)
            all_page_result.append(page_result)

            if os.path.exists(temp_path):
                os.remove(temp_path)

        final_valuation = merge_multi_page_result(all_page_result)

        return jsonify({
            "status": "Success",
            "recognition_result": final_valuation
        }), 200

    except Exception as e:
        logging.error(e)
        return jsonify({"status": "Failed", "error": str(e)}), 500

# ============================================
# IDENTITY EXTRACTION
# ============================================

def clean_student_data(raw_value, field_type):
    if not raw_value or raw_value == "Unknown":
        return "Unknown"
    
    clean_val = raw_value.strip().upper()
    
    if field_type == "roll_no":
        replacements = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1',
                        'S': '5', 'G': '6', 'B': '8', 'Z': '2'}
        for char, digit in replacements.items():
            clean_val = clean_val.replace(char, digit)
        digits = re.findall(r'\d+', clean_val)
        return "".join(digits) if digits else "Unknown"
    
    if field_type == "class":
        digits_match = re.search(r'\d+', clean_val)
        if digits_match:
            num = digits_match.group()
            if num.startswith('5') and len(num) > 1:
                return "S" + num[1:]
            return "S" + num
        return clean_val
    
    if field_type == "subject":
        if "ALORS" in clean_val or clean_val == "AL":
            return "AI"
        return clean_val
    
    return clean_val

def extract_series_identity(document_annotation):
    full_text = document_annotation.text
    details = {"name": "Unknown", "class": "Unknown", "subject": "Unknown", "roll_no": "Unknown"}
    
    patterns = {
        "name": r"Name\s*[:\-]\s*([A-Za-z\s\.]+)",
        "class": r"Class\s*[:\-]\s*([A-Za-z0-9\s]+)",
        "subject": r"Subject\s*[:\-]\s*([A-Za-z0-9\s]+)",
        "roll_no": r"Roll\s*(?:No|#)?\s*[:\-]\s*([A-Z0-9]+)"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            raw_val = match.group(1).strip().split('\n')[0]
            details[key] = clean_student_data(raw_val, key)
    
    return details

# ============================================
# BATCH EVALUATION
# ============================================

def evaluate_series_paper(student_id, answer_files, manual_roll_no, manual_class, manual_subject, exam_id=None):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            student_id.save(tmp.name)
            id_temp_path = tmp.name

        id_annotation = get_document_annotation(id_temp_path)
        student_info = extract_series_identity(id_annotation)

        if manual_roll_no and manual_roll_no != 'N/A':
            student_info["roll_no"] = manual_roll_no
        student_info['class'] = manual_class
        student_info['subject'] = manual_subject

        if os.path.exists(id_temp_path):
            os.remove(id_temp_path)

        config = {'is_handwritten': True}

        exam_data_str = request.form.get('exam_data')
        if exam_data_str:
            try:
                exam_data = json.loads(exam_data_str)
                question_types = exam_data.get('question_types', {})
                config['question_types'] = question_types
                exam_id = exam_data.get('exam_id', exam_id)
            except Exception as e:
                logging.error(f"Failed to parse exam_data: {str(e)}")
                exam_data = None
                exam_id = None
        else:
            if exam_id:
                answer_key = get_answer_key_by_id(exam_id)
                if answer_key:
                    question_types = answer_key.get('question_types', {})
                    config['question_types'] = question_types
                else:
                    exam_id = None

        all_pages_result = []
        for index, file in enumerate(answer_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                file.save(tmp.name)
                temp_path = tmp.name

            page_result = detect_and_segment_image(temp_path, debug=True, config=config)
            all_pages_result.append(page_result)

            if os.path.exists(temp_path):
                os.remove(temp_path)

        final_valuation = merge_multi_page_result(all_pages_result)

        return jsonify({
            "status": "Success",
            "student_info": student_info,
            "recognition_result": final_valuation,
            "saved_to_exam": exam_id if exam_id else None
        }), 200
        
    except Exception as e:
        logging.error(e)
        return jsonify({"status": "Failed", "error": str(e)}), 500

# ============================================
# ANSWER KEY EXTRACTION
# ============================================

def extract_answer_key_text_util(answer_key_image, answer_type):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            answer_key_image.save(tmp.name)
            temp_path = tmp.name
        
        config = {
            'default_answer_type': answer_type,
            'strict_validation': False,
            'is_handwritten': False
        }
        
        result = detect_and_segment_image(temp_path, debug=True, config=config)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {
            "status": "Success",
            "answers": result['answers'],
            "metadata": result.get('metadata', {})
        }
        
    except Exception as e:
        logging.error(f"Extract answer key error: {str(e)}")
        raise

def save_answer_key_util(data):
    try:
        exam_name = data.get('exam_name')
        class_name = data.get('class_name')
        subject = data.get('subject')
        short_questions_str = data.get('short_questions')
        long_questions_str = data.get('long_questions')
        short_marks_str = data.get('short_marks', '')
        long_marks_str = data.get('long_marks', '')
        short_answers = data.get('short_answers', {})
        long_answers = data.get('long_answers', {})
        or_groups = data.get('or_groups', [])

        if not all([exam_name, class_name, subject]):
            return {"status": "Failed", "error": "Missing required fields: exam_name, class_name, or subject"}

        exam_id = generate_exam_id(exam_name, class_name, subject)

        short_questions = parse_question_range(short_questions_str) if short_questions_str else []
        long_questions = parse_question_range(long_questions_str) if long_questions_str else []

        if not short_questions and not long_questions:
            return {"status": "Failed", "error": "Please specify at least one question range"}

        question_types = {}
        for q in short_questions:
            question_types[str(q)] = 'short'
        for q in long_questions:
            question_types[str(q)] = 'long'

        question_marks = {}
        total_marks = 0

        if short_questions and short_marks_str:
            try:
                short_marks_list = parse_marks_string(short_marks_str, len(short_questions))
                for q_num, mark in zip(short_questions, short_marks_list):
                    question_marks[str(q_num)] = mark
                    total_marks += mark
            except ValueError as e:
                return {"status": "Failed", "error": f"Short question marks error: {str(e)}"}
        elif short_questions and not short_marks_str:
            return {"status": "Failed", "error": "Short questions specified but no marks provided"}

        if long_questions and long_marks_str:
            try:
                long_marks_list = parse_marks_string(long_marks_str, len(long_questions))
                for q_num, mark in zip(long_questions, long_marks_list):
                    question_marks[str(q_num)] = mark
                    total_marks += mark
            except ValueError as e:
                return {"status": "Failed", "error": f"Long question marks error: {str(e)}"}
        elif long_questions and not long_marks_str:
            return {"status": "Failed", "error": "Long questions specified but no marks provided"}

        if total_marks <= 0:
            return {"status": "Failed", "error": "Total marks must be greater than 0"}

        processed_or_groups = []
        if or_groups:
            for group in or_groups:
                group_type = group.get('type')
                if group_type == 'single':
                    options = group.get('options', [])
                    if len(options) == 2:
                        processed_or_groups.append({'type': 'single', 'options': options})
                elif group_type == 'pair':
                    option_a = group.get('option_a', [])
                    option_b = group.get('option_b', [])
                    if option_a and option_b:
                        processed_or_groups.append({'type': 'pair', 'option_a': option_a, 'option_b': option_b})

        exam_data = {
            'exam_metadata': {
                'exam_id': exam_id,
                'exam_name': exam_name,
                'class': class_name,
                'subject': subject,
                'total_marks': total_marks,
                'created_at': str(uuid.uuid4())
            },
            'question_types': question_types,
            'question_marks': question_marks,
            'teacher_answers': {**short_answers, **long_answers},
            'or_groups': processed_or_groups,
            'student_submissions': {}
        }

        all_exam_data = load_answer_keys()
        all_exam_data[exam_id] = exam_data
        save_answer_keys(all_exam_data)

        return {
            "status": "Success",
            "exam_id": exam_id,
            "message": "Answer key saved successfully",
            "question_count": len(question_types),
            "total_marks": total_marks,
            "or_groups_count": len(processed_or_groups)
        }
        
    except Exception as e:
        logging.error(f"Save answer key error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise

# ============================================
# STUDENT SUBMISSION MANAGEMENT
# ============================================

def save_student_submission(exam_id, roll_no, student_info, answers):
    try:
        all_exam_data = load_answer_keys()
        if exam_id not in all_exam_data:
            return {"status": "Failed", "error": f"Exam {exam_id} does not exist"}
        
        exam_data = all_exam_data[exam_id]
        
        import datetime
        submission = {
            'student_info': student_info,
            'answers': answers,
            'submitted_at': datetime.datetime.now().isoformat(),
            'valuation_status': 'pending',
            'marks_awarded': None,
            'total_marks_obtained': None,
            'percentage': None
        }
        
        exam_data['student_submissions'][roll_no] = submission
        all_exam_data[exam_id] = exam_data
        save_answer_keys(all_exam_data)
        
        return {"status": "Success", "message": "Student submission saved"}
        
    except Exception as e:
        logging.error(f"Save submission error: {str(e)}")
        raise

def get_exam_with_submissions(exam_id):
    try:
        all_exam_data = load_answer_keys()
        if exam_id not in all_exam_data:
            return None
        
        exam_data = all_exam_data[exam_id]
        return {
            'exam_metadata': exam_data.get('exam_metadata', {}),
            'question_types': exam_data.get('question_types', {}),
            'question_marks': exam_data.get('question_marks', {}),
            'teacher_answers': exam_data.get('teacher_answers', {}),
            'student_submissions': exam_data.get('student_submissions', {}),
            'total_students': len(exam_data.get('student_submissions', {}))
        }
        
    except Exception as e:
        logging.error(f"Get exam error: {str(e)}")
        raise

# ============================================
# EVALUATION FUNCTIONS
# ============================================

def evaluate_student_submission(exam_id, roll_no):
    try:
        all_exam_data = load_answer_keys()
        if exam_id not in all_exam_data:
            return {"status": "Failed", "error": f"Exam {exam_id} not found"}
        
        exam_data = all_exam_data[exam_id]
        if roll_no not in exam_data.get('student_submissions', {}):
            return {"status": "Failed", "error": f"Student {roll_no} submission not found"}

        student_data = exam_data['student_submissions'][roll_no]
        teacher_answers = exam_data.get('teacher_answers', {})
        student_answers = student_data.get('answers', {})
        question_types = exam_data.get('question_types', {})
        question_marks = exam_data.get('question_marks', {})
        or_groups = exam_data.get('or_groups', [])

        marks_breakdown = {}
        total_marks_obtained = 0.0
        total_marks_possible = 0

        or_question_map = {}
        for idx, group in enumerate(or_groups):
            if group['type'] == 'single':
                for q in group['options']:
                    or_question_map[q] = idx
            elif group['type'] == 'pair':
                for q in group['option_a'] + group['option_b']:
                    or_question_map[q] = idx

        or_group_scores = {}

        for q_label, student_ans in student_answers.items():
            q_num = normalize_question_key(q_label, add_prefix=False)
            
            if q_num not in question_types:
                continue
            
            q_type = question_types[q_num]
            max_marks = question_marks[q_num]
            teacher_ans = teacher_answers.get(q_label, "")
            
            if not teacher_ans:
                continue
            
            if q_type == "short":
                score = evaluation_short_answer(
                    student_answer=student_ans,
                    teacher_answer=teacher_ans,
                    max_mark=max_marks
                )
            elif q_type == "long":
                keypoints = smart_paragraph_split(teacher_ans)
                score = evaluation_long_answer(
                    student_answer=student_ans,
                    teacher_answer=keypoints,
                    max_mark=max_marks
                )
            else:
                continue

            if q_num in or_question_map:
                group_idx = or_question_map[q_num]
                if group_idx not in or_group_scores:
                    or_group_scores[group_idx] = {}
                or_group_scores[group_idx][q_num] = {
                    'score': score,
                    'max_marks': max_marks,
                    'q_label': q_label,
                    'q_type': q_type
                }
            else:
                marks_breakdown[q_label] = {
                    "marks_obtained": score,
                    "max_marks": max_marks,
                    "question_type": q_type
                }
                total_marks_obtained += score
                total_marks_possible += max_marks

        for group_idx, group_data in or_group_scores.items():
            group = or_groups[group_idx]
            
            if group['type'] == 'single':
                best_q = max(group_data, key=lambda q: group_data[q]['score'])
                q_data = group_data[best_q]
                marks_breakdown[q_data['q_label']] = {
                    "marks_obtained": q_data['score'],
                    "max_marks": q_data['max_marks'],
                    "question_type": q_data['q_type'],
                    "or_group": True,
                    "or_type": "single",
                    "or_options": group['options'],
                    "chosen_option": best_q
                }
                total_marks_obtained += q_data['score']
                total_marks_possible += q_data['max_marks']
            
            elif group['type'] == 'pair':
                pair_a_total = sum(group_data[q]['score'] for q in group['option_a'] if q in group_data)
                pair_a_max = sum(group_data[q]['max_marks'] for q in group['option_a'] if q in group_data)
                pair_b_total = sum(group_data[q]['score'] for q in group['option_b'] if q in group_data)
                pair_b_max = sum(group_data[q]['max_marks'] for q in group['option_b'] if q in group_data)
                
                if pair_a_total >= pair_b_total:
                    chosen_pair, chosen_questions, chosen_total, chosen_max = 'a', group['option_a'], pair_a_total, pair_a_max
                else:
                    chosen_pair, chosen_questions, chosen_total, chosen_max = 'b', group['option_b'], pair_b_total, pair_b_max
                
                for q_num in chosen_questions:
                    if q_num in group_data:
                        q_data = group_data[q_num]
                        marks_breakdown[q_data['q_label']] = {
                            "marks_obtained": q_data['score'],
                            "max_marks": q_data['max_marks'],
                            "question_type": q_data['q_type'],
                            "or_group": True,
                            "or_type": "pair",
                            "or_pair_a": group['option_a'],
                            "or_pair_b": group['option_b'],
                            "chosen_pair": chosen_pair
                        }
                
                total_marks_obtained += chosen_total
                total_marks_possible += chosen_max

        percentage = (total_marks_obtained / total_marks_possible * 100) if total_marks_possible > 0 else 0

        exam_data['student_submissions'][roll_no]['marks_awarded'] = marks_breakdown
        exam_data['student_submissions'][roll_no]['total_marks_obtained'] = round(total_marks_obtained, 2)
        exam_data['student_submissions'][roll_no]['percentage'] = round(percentage, 2)
        exam_data['student_submissions'][roll_no]['valuation_status'] = 'completed'
        all_exam_data[exam_id] = exam_data
        save_answer_keys(all_exam_data)

        return {
            "status": "Success",
            "roll_no": roll_no,
            "student_name": student_data['student_info'].get('name', 'Unknown'),
            "marks_breakdown": marks_breakdown,
            "total_marks_obtained": round(total_marks_obtained, 2),
            "total_marks_possible": total_marks_possible,
            "percentage": round(percentage, 2),
            "result": "Pass" if percentage >= 40 else "Fail"
        }
        
    except Exception as e:
        logging.error(f"Evaluate submission error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return {"status": "Failed", "error": str(e)}

def evaluate_student_with_exam_data(exam_data, roll_no, student_data):
    try:
        teacher_answers = exam_data.get('teacher_answers', {})
        student_answers = student_data.get('answers', {})
        question_types = exam_data.get('question_types', {})
        question_marks = exam_data.get('question_marks', {})
        or_groups = exam_data.get('or_groups', [])
        
        marks_breakdown = {}
        total_marks_obtained = 0.0
        
        # Calculate exam total considering OR groups
        or_question_numbers = set()
        for group in or_groups:
            if group['type'] == 'single':
                or_question_numbers.update(str(q) for q in group['options'])
            elif group['type'] == 'pair':
                or_question_numbers.update(str(q) for q in group.get('option_a', []))
                or_question_numbers.update(str(q) for q in group.get('option_b', []))
        
        exam_total_marks = 0
        
        for q_num, marks in question_marks.items():
            if str(q_num) not in or_question_numbers:
                exam_total_marks += int(marks)
        
        for group in or_groups:
            if group['type'] == 'single':
                first_option = str(group['options'][0])
                exam_total_marks += int(question_marks.get(first_option, 0))
            elif group['type'] == 'pair':
                for q in group.get('option_a', []):
                    exam_total_marks += int(question_marks.get(str(q), 0))
        
        or_question_map = {}
        for idx, group in enumerate(or_groups):
            if group['type'] == 'single':
                for q in group['options']:
                    or_question_map[str(q)] = idx
            elif group['type'] == 'pair':
                for q in group.get('option_a', []) + group.get('option_b', []):
                    or_question_map[str(q)] = idx
        
        or_group_scores = {}
        
        for q_label, student_ans in student_answers.items():
            q_num = normalize_question_key(q_label, add_prefix=False)
            
            if q_num not in question_types:
                continue
            
            q_type = question_types[q_num]
            max_marks = question_marks.get(q_num, 0)
            teacher_ans = teacher_answers.get(q_label, "")
            
            if not teacher_ans:
                continue
            
            if q_type == "short":
                score = evaluation_short_answer(
                    student_answer=student_ans,
                    teacher_answer=teacher_ans,
                    max_mark=max_marks
                )
            elif q_type == "long":
                keypoints = smart_paragraph_split(teacher_ans)
                score = evaluation_long_answer(
                    student_answer=student_ans,
                    teacher_answer=keypoints,
                    max_mark=max_marks
                )
            else:
                continue
            
            if q_num in or_question_map:
                group_idx = or_question_map[q_num]
                if group_idx not in or_group_scores:
                    or_group_scores[group_idx] = {}
                or_group_scores[group_idx][q_num] = {
                    'score': score,
                    'max_marks': max_marks,
                    'q_label': q_label,
                    'q_type': q_type
                }
            else:
                marks_breakdown[q_label] = {
                    "marks_obtained": score,
                    "max_marks": max_marks,
                    "question_type": q_type
                }
                total_marks_obtained += score
        
        for group_idx, group_data in or_group_scores.items():
            group = or_groups[group_idx]
            
            if group['type'] == 'single':
                best_q = max(group_data.items(), key=lambda x: x[1]['score'])[0]
                q_data = group_data[best_q]
                marks_breakdown[q_data['q_label']] = {
                    "marks_obtained": q_data['score'],
                    "max_marks": q_data['max_marks'],
                    "question_type": q_data['q_type'],
                    "or_group": True,
                    "or_type": "single",
                    "or_options": group['options'],
                    "chosen_option": best_q
                }
                total_marks_obtained += q_data['score']
            
            elif group['type'] == 'pair':
                pair_a_total = sum(group_data[q]['score'] for q in group.get('option_a', []) if q in group_data)
                pair_b_total = sum(group_data[q]['score'] for q in group.get('option_b', []) if q in group_data)
                
                if pair_a_total >= pair_b_total:
                    chosen_pair = 'a'
                    chosen_questions = group.get('option_a', [])
                    chosen_total = pair_a_total
                else:
                    chosen_pair = 'b'
                    chosen_questions = group.get('option_b', [])
                    chosen_total = pair_b_total
                
                for q_num in chosen_questions:
                    if q_num in group_data:
                        q_data = group_data[q_num]
                        marks_breakdown[q_data['q_label']] = {
                            "marks_obtained": q_data['score'],
                            "max_marks": q_data['max_marks'],
                            "question_type": q_data['q_type'],
                            "or_group": True,
                            "or_type": "pair",
                            "or_pair_a": group.get('option_a', []),
                            "or_pair_b": group.get('option_b', []),
                            "chosen_pair": chosen_pair
                        }
                
                total_marks_obtained += chosen_total
        
        percentage = (total_marks_obtained / exam_total_marks * 100) if exam_total_marks > 0 else 0
        
        return {
            "status": "Success",
            "roll_no": roll_no,
            "student_name": student_data.get('student_info', {}).get('name', 'Unknown'),
            "marks_breakdown": marks_breakdown,
            "total_marks_obtained": round(total_marks_obtained, 2),
            "total_marks_possible": exam_total_marks,
            "percentage": round(percentage, 2),
            "result": "Pass" if percentage >= 40 else "Fail"
        }
        
    except Exception as e:
        logging.error(f"Evaluate student error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return {"status": "Failed", "error": str(e)}
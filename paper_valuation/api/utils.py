import re
import json
import uuid
import os
import tempfile
from flask import jsonify
from paper_valuation.logging.logger import logging
from paper_valuation.api.vision_segmentation import detect_and_segment_image, get_document_annotation


ANSWER_KEYS_FILE = 'answer_keys.json'

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



def merge_multi_page_result(all_pages_list):
    merged_answers = {}
    last_q_label = None
    
    for page_index, page in enumerate(all_pages_list):
        answers = page.get('answers', {})
        
        if 'UNLABELED_CONTINUATION' in answers:
            unlabeled_text = answers.pop('UNLABELED_CONTINUATION')
            
            if last_q_label and last_q_label in merged_answers:
                merged_answers[last_q_label] += " " + unlabeled_text.strip()
                #print(f"⚠️  Page {page_index + 1}: Unlabeled continuation appended to {last_q_label}")
            else:
                if 'Q1' in merged_answers:
                    merged_answers['Q1'] = unlabeled_text.strip() + " " + merged_answers['Q1']
                else:
                    merged_answers['Q1'] = unlabeled_text.strip()
                last_q_label = 'Q1'
                #print(f"⚠️  Page {page_index + 1}: Unlabeled page assigned to Q1 by default")

        for q_label, text in answers.items():
            if q_label in merged_answers:
                merged_answers[q_label] += " " + text.strip()
                print(f"✅ Page {page_index + 1}: {q_label} continuation merged")
            else:
                merged_answers[q_label] = text.strip()
                print(f"✅ Page {page_index + 1}: Added {q_label}")
            
            last_q_label = q_label
    
    sorted_answers = dict(sorted(
        merged_answers.items(),
        key=lambda x: int(re.search(r'\d+', x[0]).group()) if re.search(r'\d+', x[0]) else 0
    ))
    
    print(f"\n📊 Final Merge Summary: {len(sorted_answers)} questions across {len(all_pages_list)} pages")
    return {"answers": sorted_answers, "total_pages": len(all_pages_list)}



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
                
            logging.info(f"Processing Page {index + 1}: {file.filename} -> {temp_path}")
            
            page_result = detect_and_segment_image(temp_path, debug=True, config=config)
            all_page_result.append(page_result)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logging.info(f"Cleaned up Page {index + 1} temp file.")
                
        final_valuation = merge_multi_page_result(all_page_result)
        
        return jsonify({
            "status": "Success",
            "recognition_result": final_valuation
        }), 200

    except Exception as e:
        logging.error(e)
        return jsonify({"status": "Failed", "error": str(e)}), 500



def clean_student_data(raw_value, field_type):
    if not raw_value or raw_value == "Unknown":
        return "Unknown"

    clean_val = raw_value.strip().upper()

    if field_type == "roll_no":
        replacements = {
            'O': '0', 'D': '0', 'Q': '0', 
            'I': '1', 'L': '1', 
            'S': '5', 'G': '6', 'B': '8',
            'Z': '2'
        }
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
    details = {
        "name": "Unknown",
        "class": "Unknown",
        "subject": "Unknown",
        "roll_no": "Unknown"
    }

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



def evaluate_series_paper(student_id, answer_files, manual_roll_no, manual_class, manual_subject, exam_id=None):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            student_id.save(tmp.name)
            id_temp_path = tmp.name
            
        logging.info(f"Extracting identity from: {student_id.filename}")
        id_annotation = get_document_annotation(id_temp_path)
        student_info = extract_series_identity(id_annotation)
        
        if manual_roll_no and manual_roll_no != 'N/A':
            student_info["roll_no"] = manual_roll_no
            
        student_info['class'] = manual_class
        student_info['subject'] = manual_subject
        
        if os.path.exists(id_temp_path):
            os.remove(id_temp_path)
        
        config = {
            'is_handwritten': True  
        }
        
        if exam_id:
            answer_key = get_answer_key_by_id(exam_id)
            if answer_key:
                question_types = answer_key.get('question_types', {})
                config['question_types'] = question_types
                
                exam_name = answer_key.get('exam_metadata', {}).get('exam_name', answer_key.get('exam_name', 'Unknown'))
                logging.info(f"✅ Loaded answer key: {exam_name}")
                logging.info(f"   Question types: {len([q for q in question_types.values() if q == 'short'])} short, "
                           f"{len([q for q in question_types.values() if q == 'long'])} long")
            else:
                logging.warning(f"⚠️ Exam ID {exam_id} not found. Using default settings.")
        
        all_pages_result = []
        for index, file in enumerate(answer_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                file.save(tmp.name)
                temp_path = tmp.name
            
            logging.info(f"Processing Answer Page {index + 1} for {student_info['name']} (HANDWRITTEN)")
            
            page_result = detect_and_segment_image(temp_path, debug=True, config=config)
            all_pages_result.append(page_result)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)

        final_valuation = merge_multi_page_result(all_pages_result)
        
        if exam_id and student_info['roll_no'] != 'Unknown':
            try:
                save_result = save_student_submission(
                    exam_id=exam_id,
                    roll_no=student_info['roll_no'],
                    student_info=student_info,
                    answers=final_valuation['answers']
                )
                logging.info(f"✅ Saved student {student_info['roll_no']} to exam {exam_id}")
            except Exception as save_error:
                logging.error(f"⚠️ Failed to save student submission: {str(save_error)}")
        else:
            if not exam_id:
                logging.warning("⚠️ No exam_id provided - student submission not saved to storage")
            if student_info['roll_no'] == 'Unknown':
                logging.warning("⚠️ Roll number is Unknown - student submission not saved to storage")
            
        return jsonify({
            "status": "Success",
            "student_info": student_info,
            "recognition_result": final_valuation,
            "saved_to_exam": exam_id if exam_id else None  # 🆕 NEW
        }), 200
    except Exception as e:
        logging.error(e)
        return jsonify({"status": "Failed", "error": str(e)}), 500



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
        logging.info(f"✅ Successfully extracted {len(result['answers'])} answers as '{answer_type}' type (PRINTED)")
        
        return {
            "status": "Success",
            "answers": result['answers'],
            "metadata": result.get('metadata', {})
        }
    except Exception as e:
        logging.error(f"Error in extract_answer_key_text_util: {str(e)}")
        raise



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
            raise ValueError(f"Invalid marks format: '{marks_str}'. Use single number or comma-separated numbers.")
    
    marks_parts = [part.strip() for part in marks_str.split(',')]
    
    try:
        marks_list = [int(mark) for mark in marks_parts]
    except ValueError:
        raise ValueError(f"Invalid marks format: '{marks_str}'. All values must be numbers.")
    
    if any(mark <= 0 for mark in marks_list):
        raise ValueError("All marks must be positive numbers")
    
    if len(marks_list) != question_count:
        raise ValueError(
            f"Marks count mismatch: provided {len(marks_list)} marks but have {question_count} questions. "
            f"Use single number for uniform marks or provide exact count."
        )
    return marks_list



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
        
        if not all([exam_name, class_name, subject]):
            return {"status": "Failed", "error": "Missing required fields: exam_name, class_name, or subject"}
        
        logging.info(f"💾 Saving answer key for: {exam_name} ({class_name} - {subject})")
        
        exam_id = generate_exam_id(exam_name, class_name, subject)
        
        short_questions = parse_question_range(short_questions_str) if short_questions_str else []
        long_questions = parse_question_range(long_questions_str) if long_questions_str else []
        
        if not short_questions and not long_questions:
            return {"status": "Failed", "error": "Please specify at least one question range (short or long)"}
        
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
                logging.info(f"   ✅ Short questions: {len(short_questions)} questions, marks: {short_marks_list}")
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
                logging.info(f"   ✅ Long questions: {len(long_questions)} questions, marks: {long_marks_list}")
            except ValueError as e:
                return {"status": "Failed", "error": f"Long question marks error: {str(e)}"}
        elif long_questions and not long_marks_str:
            return {"status": "Failed", "error": "Long questions specified but no marks provided"}
        
        if total_marks <= 0:
            return {"status": "Failed", "error": "Total marks must be greater than 0"}
        
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
            'student_submissions': {}  
        }
        
        all_exam_data = load_answer_keys()  
        all_exam_data[exam_id] = exam_data
        save_answer_keys(all_exam_data)
        
        logging.info(f"✅ Answer key saved: {exam_id}")
        logging.info(f"   📊 {len(short_questions)} short + {len(long_questions)} long = {len(question_types)} total questions")
        logging.info(f"   💯 Total marks: {total_marks}")
        logging.info(f"   👥 Student submissions structure created (empty)")
        
        return {
            "status": "Success",
            "exam_id": exam_id,
            "message": "Answer key saved successfully",
            "question_count": len(question_types),
            "total_marks": total_marks
        }
        
    except Exception as e:
        logging.error(f"Error in save_answer_key_util: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise
    
    
    
def save_student_submission(exam_id, roll_no, student_info, answers):
    try:
        all_exam_data = load_answer_keys()
        
        if exam_id not in all_exam_data:
            logging.error(f"❌ Exam ID {exam_id} not found in storage")
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
        
        logging.info(f"✅ Saved submission for Roll No: {roll_no} to Exam: {exam_id}")
        
        return {"status": "Success", "message": "Student submission saved"}
        
    except Exception as e:
        logging.error(f"Error saving student submission: {str(e)}")
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
        logging.error(f"Error retrieving exam data: {str(e)}")
        raise
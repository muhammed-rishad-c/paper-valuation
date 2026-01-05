import re
import json
import uuid
import os
import tempfile
from flask import jsonify
from paper_valuation.logging.logger import logging
from paper_valuation.api.vision_segmentation import detect_and_segment_image, get_document_annotation

# ============================================
# ANSWER KEY STORAGE CONFIGURATION
# ============================================

ANSWER_KEYS_FILE = 'answer_keys.json'

def load_answer_keys():
    """Load all answer keys from storage"""
    if os.path.exists(ANSWER_KEYS_FILE):
        with open(ANSWER_KEYS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_answer_keys(answer_keys):
    """Save answer keys to storage"""
    with open(ANSWER_KEYS_FILE, 'w') as f:
        json.dump(answer_keys, f, indent=2)

def get_answer_key_by_id(exam_id):
    """Get specific answer key by exam_id"""
    all_keys = load_answer_keys()
    return all_keys.get(exam_id, None)

def generate_exam_id(exam_name, class_name, subject):
    """Generate a unique exam ID"""
    base = f"{subject}_{class_name}_{exam_name.replace(' ', '_')}"
    unique_suffix = str(uuid.uuid4())[:8].upper()
    return f"{base}_{unique_suffix}"

def parse_question_range(range_str):
    """
    Parse question range string into list of question numbers
    Examples: "1-10" -> [1,2,3,...,10]
              "1,2,3,5" -> [1,2,3,5]
    """
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

# ============================================
# MULTI-PAGE MERGE LOGIC
# ============================================

def merge_multi_page_result(all_pages_list):
    """Merge results from multiple pages into single answer set"""
    merged_answers = {}
    last_q_label = None
    
    for page_index, page in enumerate(all_pages_list):
        answers = page.get('answers', {})
        
        if 'UNLABELED_CONTINUATION' in answers:
            unlabeled_text = answers.pop('UNLABELED_CONTINUATION')
            
            if last_q_label and last_q_label in merged_answers:
                merged_answers[last_q_label] += " " + unlabeled_text.strip()
                print(f"⚠️  Page {page_index + 1}: Unlabeled continuation appended to {last_q_label}")
            else:
                if 'Q1' in merged_answers:
                    merged_answers['Q1'] = unlabeled_text.strip() + " " + merged_answers['Q1']
                else:
                    merged_answers['Q1'] = unlabeled_text.strip()
                last_q_label = 'Q1'
                print(f"⚠️  Page {page_index + 1}: Unlabeled page assigned to Q1 by default")

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

# ============================================
# INDIVIDUAL EVALUATION
# ============================================

def evaluate_paper_individual(files, config=None):
    """Evaluate individual student paper (multiple pages)"""
    try:
        all_page_result = []
        
        for index, file in enumerate(files):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                file.save(tmp.name)
                temp_path = tmp.name
                
            logging.info(f"Processing Page {index + 1}: {file.filename} -> {temp_path}")
            
            # Pass config to segmentation
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

# ============================================
# IDENTITY EXTRACTION
# ============================================

def clean_student_data(raw_value, field_type):
    """Cleans OCR noise and recovers misread handwritten characters"""
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
    """Extract student identity from cover page"""
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

# ============================================
# SERIES EVALUATION
# ============================================

def evaluate_series_paper(student_id, answer_files, manual_roll_no, manual_class, manual_subject, exam_id=None):
    """Evaluate series paper with answer key support"""
    try:
        # Extract identity
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
        
        # Load answer key configuration if exam_id provided
        config = None
        if exam_id:
            answer_key = get_answer_key_by_id(exam_id)
            if answer_key:
                question_types = answer_key['question_types']
                config = {'question_types': question_types}
                logging.info(f"✅ Loaded answer key: {answer_key['exam_name']}")
                logging.info(f"   Question types: {len([q for q in question_types.values() if q == 'short'])} short, "
                           f"{len([q for q in question_types.values() if q == 'long'])} long")
            else:
                logging.warning(f"⚠️ Exam ID {exam_id} not found. Using default settings.")
        
        # Process answer pages
        all_pages_result = []
        for index, file in enumerate(answer_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                file.save(tmp.name)
                temp_path = tmp.name
            
            logging.info(f"Processing Answer Page {index + 1} for {student_info['name']}")
            
            # Pass config with question types to segmentation
            page_result = detect_and_segment_image(temp_path, debug=True, config=config)
            all_pages_result.append(page_result)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Merge all answers
        final_valuation = merge_multi_page_result(all_pages_result)
            
        return jsonify({
            "status": "Success",
            "student_info": student_info,
            "recognition_result": final_valuation
        }), 200
        
    except Exception as e:
        logging.error(e)
        return jsonify({"status": "Failed", "error": str(e)}), 500

# ============================================
# ANSWER KEY EXTRACTION
# ============================================

def extract_answer_key_text_util(answer_key_image, answer_type):
    """Extract text from answer key image"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            answer_key_image.save(tmp.name)
            temp_path = tmp.name
        
        # ✅ FIX: Pass default_answer_type to tell system how to format ALL questions on this page
        config = {
            'default_answer_type': answer_type,  # 'short' or 'long' from upload context
            'strict_validation': False
        }
        
        result = detect_and_segment_image(temp_path, debug=True, config=config)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        logging.info(f"✅ Successfully extracted {len(result['answers'])} answers as '{answer_type}' type")
        
        return {
            "status": "Success",
            "answers": result['answers'],
            "metadata": result.get('metadata', {})
        }
        
    except Exception as e:
        logging.error(f"Error in extract_answer_key_text_util: {str(e)}")
        raise

# ============================================
# ANSWER KEY SAVE
# ============================================
def save_answer_key_util(data):
    """Save complete answer key with metadata"""
    try:
        exam_name = data.get('exam_name')
        class_name = data.get('class_name')
        subject = data.get('subject')
        short_questions_str = data.get('short_questions')
        long_questions_str = data.get('long_questions')
        
        # ✅ FIX: No json.loads() needed - data is already parsed by Flask
        short_answers = data.get('short_answers', {})
        long_answers = data.get('long_answers', {})
        
        if not all([exam_name, class_name, subject]):
            return {"status": "Failed", "error": "Missing required fields"}
        
        logging.info(f"💾 Saving answer key for: {exam_name} ({class_name} - {subject})")
        
        exam_id = generate_exam_id(exam_name, class_name, subject)
        
        short_questions = parse_question_range(short_questions_str) if short_questions_str else []
        long_questions = parse_question_range(long_questions_str) if long_questions_str else []
        
        question_types = {}
        for q in short_questions:
            question_types[str(q)] = 'short'
        for q in long_questions:
            question_types[str(q)] = 'long'
        
        answer_key = {
            'exam_id': exam_id,
            'exam_name': exam_name,
            'class': class_name,
            'subject': subject,
            'short_questions_range': short_questions_str,
            'long_questions_range': long_questions_str,
            'question_types': question_types,
            'short_answers': short_answers,
            'long_answers': long_answers
        }
        
        all_answer_keys = load_answer_keys()
        all_answer_keys[exam_id] = answer_key
        save_answer_keys(all_answer_keys)
        
        logging.info(f"✅ Answer key saved: {exam_id}")
        logging.info(f"   {len(short_questions)} short, {len(long_questions)} long questions")
        
        return {
            "status": "Success",
            "exam_id": exam_id,
            "message": "Answer key saved successfully",
            "question_count": len(question_types)
        }
        
    except Exception as e:
        logging.error(f"Error in save_answer_key_util: {str(e)}")
        raise
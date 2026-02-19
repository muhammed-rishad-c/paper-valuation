import os
import sys
import traceback
import json
import uuid
from flask import Flask, request, jsonify

from paper_valuation.logging.logger import logging
from paper_valuation.api.utils import (
    evaluate_paper_individual, 
    evaluate_series_paper,
    extract_answer_key_text_util,
    save_answer_key_util,
    load_answer_keys,
    get_answer_key_by_id,
    get_exam_with_submissions,
    evaluate_student_submission,
    evaluate_student_with_exam_data 
)

app = Flask(__name__)

    
@app.route('/api/evaluate_individual', methods=['POST'])
def evaluate_individual_endpoint():
    """
    Individual paper evaluation WITH answer key
    - Receives: exam_id + paper_images
    - Returns: Evaluated results with marks breakdown
    """
    try:
        exam_id = request.form.get('exam_id')
        files = request.files.getlist('paper_images')
        
        if not exam_id:
            logging.error("No exam_id provided")
            return jsonify({"status": "Failed", "error": "Exam ID is required"}), 400
        
        if not files or files[0].filename == '':
            logging.error("No images found in the request")
            return jsonify({"status": "Failed", "error": "No images provided"}), 400
        
        logging.info("="*70)
        logging.info(f"Individual Evaluation with Answer Key")
        logging.info(f"Exam ID: {exam_id}")
        logging.info(f"Processing {len(files)} pages")
        logging.info("="*70)
        
        # Load answer key
        answer_key = get_answer_key_by_id(exam_id)
        if not answer_key:
            return jsonify({
                "status": "Failed", 
                "error": f"Answer key not found for exam {exam_id}"
            }), 404
        
        exam_name = answer_key.get('exam_metadata', {}).get('exam_name', answer_key.get('exam_name', 'Unknown'))
        logging.info(f"✅ Loaded answer key: {exam_name}")
        
        # Extract answers with question type configuration
        question_types = answer_key.get('question_types', {})
        config = {
            'is_handwritten': True,
            'question_types': question_types
        }
        
        results = evaluate_paper_individual(files, config=config)
        result_data = results[0].get_json()
        
        if result_data.get('status') != 'Success':
            return results[0]
        
        # Get extracted answers
        student_answers = result_data['recognition_result']['answers']
        
        logging.info(f"📝 Extracted {len(student_answers)} answers from student paper")
        
        # Create a temporary student submission structure for evaluation
        temp_submission = {
            'answers': student_answers,
            'student_info': {
                'name': 'Individual Submission',
                'roll_no': 'N/A'
            }
        }
        
        # Evaluate using existing logic
        teacher_answers = answer_key.get('teacher_answers', {})
        question_marks = answer_key.get('question_marks', {})
        or_groups = answer_key.get('or_groups', [])
        
        from paper_valuation.api.utils import (
            evaluation_short_answer,
            evaluation_long_answer,
            normalize_question_key
        )
        from paper_valuation.components.valuation import smart_paragraph_split
        
        marks_breakdown = {}
        total_marks_obtained = 0.0
        total_marks_possible = 0
        
        # Track OR questions
        or_question_map = {}
        for idx, group in enumerate(or_groups):
            if group['type'] == 'single':
                for q in group['options']:
                    or_question_map[q] = idx
            elif group['type'] == 'pair':
                for q in group['option_a'] + group['option_b']:
                    or_question_map[q] = idx
        
        or_group_scores = {}
        
        # Evaluate each question
        for q_label, student_ans in student_answers.items():
            q_num = normalize_question_key(q_label, add_prefix=False)
            
            if q_num not in question_types:
                logging.warning(f"⚠️ Question {q_label} not in answer key, skipping")
                continue
            
            q_type = question_types[q_num]
            max_marks = question_marks.get(q_num, 0)
            teacher_ans = teacher_answers.get(q_label, "")
            
            if not teacher_ans:
                logging.warning(f"⚠️ No teacher answer for {q_label}, skipping")
                continue
            
            # Evaluate
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
            
            logging.info(f"✅ {q_label} ({q_type}): {score}/{max_marks}")
            
            # Handle OR groups
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
        
        # Process OR groups (take best score)
        for group_idx, group_data in or_group_scores.items():
            group = or_groups[group_idx]
            
            if group['type'] == 'single':
                best_q = None
                best_score = -1
                
                for q_num, data in group_data.items():
                    if data['score'] > best_score:
                        best_score = data['score']
                        best_q = q_num
                
                if best_q:
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
                    chosen_pair = 'a'
                    chosen_questions = group['option_a']
                    chosen_total = pair_a_total
                    chosen_max = pair_a_max
                else:
                    chosen_pair = 'b'
                    chosen_questions = group['option_b']
                    chosen_total = pair_b_total
                    chosen_max = pair_b_max
                
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
        
        # Calculate percentage
        percentage = (total_marks_obtained / total_marks_possible * 100) if total_marks_possible > 0 else 0
        
        logging.info(f"✅ Evaluation completed: {total_marks_obtained}/{total_marks_possible} ({percentage:.2f}%)")
        
        return jsonify({
            "status": "Success",
            "exam_info": {
                "exam_id": exam_id,
                "exam_name": exam_name,
                "class": answer_key.get('exam_metadata', {}).get('class', 'Unknown'),
                "subject": answer_key.get('exam_metadata', {}).get('subject', 'Unknown')
            },
            "evaluation_result": {
                "marks_breakdown": marks_breakdown,
                "total_marks_obtained": round(total_marks_obtained, 2),
                "total_marks_possible": total_marks_possible,
                "percentage": round(percentage, 2),
                "result": "Pass" if percentage >= 40 else "Fail"
            },
            "recognition_result": result_data['recognition_result']
        }), 200

    except Exception as e:
        error_message = f"Critical System Error: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_message)
        return jsonify({"status": "Failed", "error": str(e)}), 500



@app.route('/api/extract_answer_key_text', methods=['POST'])
def extract_answer_key_text():
    try:
        if 'answer_key_image' not in request.files:
            return jsonify({"status": "Failed", "error": "No image file provided."}), 400
        
        answer_key_image = request.files['answer_key_image']
        answer_type = request.form.get('answer_type', 'short')
        
        logging.info(f"📄 Extracting {answer_type} answer key from: {answer_key_image.filename}")
        
        result = extract_answer_key_text_util(answer_key_image, answer_type)
        return jsonify(result), 200
        
    except Exception as e:
        logging.error(f"Error extracting answer key: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"status": "Failed", "error": str(e)}), 500
    
    

@app.route('/api/save_answer_key', methods=['POST'])
def save_answer_key():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"status": "Failed", "error": "No JSON data received"}), 400
        
        logging.info(f"Received data type: {type(data)}")
        logging.info(f"Data keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")
        logging.info(f"💾 Saving answer key for: {data.get('exam_name')}")
        
        result = save_answer_key_util(data)
        return jsonify(result), 200
        
    except Exception as e:
        import traceback
        logging.error(f"Error saving answer key: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"status": "Failed", "error": str(e)}), 500
    
     

@app.route('/api/get_answer_key/<exam_id>', methods=['GET'])
def get_answer_key(exam_id):
    try:
        result = get_answer_key_by_id(exam_id)
        if result:
            return jsonify({"status": "Success", "answer_key": result}), 200
        else:
            return jsonify({"status": "Failed", "error": "Answer key not found"}), 404
        
    except Exception as e:
        logging.error(f"Error retrieving answer key: {str(e)}")
        return jsonify({"status": "Failed", "error": str(e)}), 500
    
    

@app.route('/api/list_answer_keys', methods=['GET'])
def list_answer_keys():
    try:
        all_answer_keys = load_answer_keys()
        answer_key_list = []
        for key, value in all_answer_keys.items():
            if 'exam_metadata' in value:
                answer_key_list.append({
                    'exam_id': key,
                    'exam_name': value['exam_metadata'].get('exam_name', 'Unknown'),
                    'class': value['exam_metadata'].get('class', 'Unknown'),
                    'subject': value['exam_metadata'].get('subject', 'Unknown')
                })
            else:
                answer_key_list.append({
                    'exam_id': key,
                    'exam_name': value.get('exam_name', 'Unknown'),
                    'class': value.get('class', 'Unknown'),
                    'subject': value.get('subject', 'Unknown')
                })
        
        logging.info(f"📋 Returning {len(answer_key_list)} answer keys")
        
        return jsonify({
            "status": "Success",
            "answer_keys": answer_key_list
        }), 200
        
    except Exception as e:
        logging.error(f"Error listing answer keys: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({"status": "Failed", "error": str(e)}), 500




@app.route('/api/seriesBundleEvaluate', methods=['POST'])
def evaluate_series_batch_handler():
    manual_roll_no = request.form.get('manual_roll_no', 'N/A')
    manual_subject = request.form.get('manual_subject', 'N/A')
    manual_class = request.form.get('manual_class', 'N/A')
    exam_id = request.form.get('exam_id', None)
     
    logging.info(f"{'='*20} 🆕 NEW BATCH REQUEST {'='*20}")
    logging.info(f"Context: Student Roll No - {manual_roll_no}")
    logging.info(f"Exam ID: {exam_id if exam_id else 'Not provided (using defaults)'}")

    try:
        id_file = request.files.get('identity_page')
        answer_files = request.files.getlist('paper_images')
        
        logging.info(f"Payload Details: ID Page: {id_file.filename if id_file else 'MISSING'}, "
                     f"Answer Pages Count: {len(answer_files)}")

        if not id_file:
            error_msg = f"Validation Error for Roll No {manual_roll_no}: Identity page is missing."
            logging.error(error_msg)
            return jsonify({"status": "Failed", "error": "Identity page is required."}), 400
        
        if not answer_files:
            logging.warning(f"No answer pages provided for Roll No {manual_roll_no}.")

        logging.info(f"Starting paper evaluation logic for Student: {manual_roll_no}...")
        
        result = evaluate_series_paper(
            id_file, 
            answer_files, 
            manual_roll_no=manual_roll_no,
            manual_class=manual_class,
            manual_subject=manual_subject,
            exam_id=exam_id
        )
        logging.info(f"✅ Successfully processed evaluation for Roll No {manual_roll_no}.")
        return result
        
    except FileNotFoundError as fnf:
        logging.error(f"📁 File Error for Roll No {manual_roll_no}: {str(fnf)}")
        return jsonify({"status": "Failed", "error": "Internal file handling error."}), 500

    except Exception as e:
        detailed_error = traceback.format_exc()
        logging.critical(f"🔥 CRITICAL SYSTEM ERROR for Roll No {manual_roll_no}:\n"
                         f"Error Type: {type(e).__name__}\n"
                         f"Message: {str(e)}\n"
                         f"Traceback:\n{detailed_error}")
        
        return jsonify({
            "status": "Failed", 
            "error": "Internal Processing Error",
            "details": str(e) if app.debug else "Contact Administrator" 
        }), 500




@app.route('/api/get_exam_data/<exam_id>', methods=['GET'])
def get_complete_exam_data(exam_id):
    try:
        exam_data = get_exam_with_submissions(exam_id)
        
        if not exam_data:
            return jsonify({
                "status": "Failed", 
                "error": f"Exam {exam_id} not found"
            }), 404
        
        return jsonify({
            "status": "Success",
            "exam_data": exam_data
        }), 200
        
    except Exception as e:
        logging.error(f"Error retrieving complete exam data: {str(e)}")
        return jsonify({"status": "Failed", "error": str(e)}), 500
    
@app.route('/api/evaluate_exam/<exam_id>', methods=['POST'])
def evaluate_all_students_handler(exam_id):
    """
    Evaluate ALL students' submissions for an exam
    
    URL Parameters:
        exam_id: The exam identifier
        
    Returns:
        JSON with evaluation results for all students
    """
    try:
        logging.info(f"{'='*20} 📝 BATCH EVALUATION REQUEST {'='*20}")
        logging.info(f"Exam ID: {exam_id}")
        
        # Load exam data
        exam_data = get_exam_with_submissions(exam_id)
        
        if not exam_data:
            return jsonify({
                "status": "Failed",
                "error": f"Exam {exam_id} not found"
            }), 404
        
        student_submissions = exam_data.get('student_submissions', {})
        
        if not student_submissions:
            return jsonify({
                "status": "Failed",
                "error": "No student submissions found for this exam"
            }), 404
        
        logging.info(f"📊 Found {len(student_submissions)} students to evaluate")
        
        # Evaluate each student
        evaluation_results = []
        successful = 0
        failed = 0
        
        for roll_no in student_submissions.keys():
            logging.info(f"   Evaluating Roll No: {roll_no}...")
            
            result = evaluate_student_submission(exam_id, roll_no)
            
            if result['status'] == 'Success':
                evaluation_results.append(result)
                successful += 1
                logging.info(f"   ✅ Roll No {roll_no}: {result['total_marks_obtained']}/{result['total_marks_possible']} ({result['percentage']}%)")
            else:
                failed += 1
                logging.error(f"   ❌ Roll No {roll_no}: {result.get('error')}")
                evaluation_results.append({
                    "status": "Failed",
                    "roll_no": roll_no,
                    "error": result.get('error')
                })
        
        logging.info(f"✅ Batch evaluation completed: {successful} successful, {failed} failed")
        
        return jsonify({
            "status": "Success",
            "exam_id": exam_id,
            "exam_name": exam_data['exam_metadata'].get('exam_name'),
            "total_students": len(student_submissions),
            "evaluated_successfully": successful,
            "evaluation_failed": failed,
            "results": evaluation_results
        }), 200
        
    except Exception as e:
        logging.error(f"Error in evaluate_all_students_handler: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({"status": "Failed", "error": str(e)}), 500
    
@app.route('/api/evaluate_student/<exam_id>/<roll_no>', methods=['POST'])
def evaluate_single_student_handler(exam_id, roll_no):
    """
    Evaluate a SINGLE student's submission (optional endpoint)
    
    Useful for re-evaluation or checking one student
    """
    try:
        logging.info(f"📝 Single student evaluation: Exam {exam_id}, Roll {roll_no}")
        
        result = evaluate_student_submission(exam_id, roll_no)
        
        if result['status'] == 'Success':
            return jsonify(result), 200
        else:
            return jsonify(result), 404
            
    except Exception as e:
        logging.error(f"Error evaluating student: {str(e)}")
        return jsonify({"status": "Failed", "error": str(e)}), 500
    
@app.route('/api/evaluate_exam_with_data', methods=['POST'])
def evaluate_exam_with_complete_data():
    """
    Evaluate ALL students using exam data sent from Node.js
    (Does NOT look up in answer_keys.json)
    """
    try:
        # Get complete exam data from request body
        exam_data = request.get_json()
        
        if not exam_data:
            return jsonify({
                "status": "Failed",
                "error": "No exam data received"
            }), 400
        
        exam_id = exam_data.get('exam_id')
        student_submissions = exam_data.get('student_submissions', {})
        
        logging.info(f"{'='*20} 📝 BATCH EVALUATION (PostgreSQL Data) {'='*20}")
        logging.info(f"Exam ID: {exam_id}")
        logging.info(f"Exam Name: {exam_data.get('exam_name')}")
        
        if not student_submissions:
            return jsonify({
                "status": "Failed",
                "error": "No student submissions found"
            }), 404
        
        logging.info(f"📊 Found {len(student_submissions)} students to evaluate")
        
        # Evaluate each student using the provided exam data
        evaluation_results = []
        successful = 0
        failed = 0
        
        for roll_no, student_data in student_submissions.items():
            logging.info(f"   Evaluating Roll No: {roll_no}...")
            
            # Call evaluation function with complete exam data
            result = evaluate_student_with_exam_data(exam_data, roll_no, student_data)
            
            if result['status'] == 'Success':
                evaluation_results.append(result)
                successful += 1
                logging.info(f"   ✅ Roll No {roll_no}: {result['total_marks_obtained']}/{result['total_marks_possible']} ({result['percentage']}%)")
            else:
                failed += 1
                logging.error(f"   ❌ Roll No {roll_no}: {result.get('error')}")
                evaluation_results.append({
                    "status": "Failed",
                    "roll_no": roll_no,
                    "error": result.get('error')
                })
        
        logging.info(f"✅ Batch evaluation completed: {successful} successful, {failed} failed")
        
        return jsonify({
            "status": "Success",
            "exam_id": exam_id,
            "exam_name": exam_data.get('exam_name'),
            "total_students": len(student_submissions),
            "evaluated_successfully": successful,
            "evaluation_failed": failed,
            "results": evaluation_results
        }), 200
        
    except Exception as e:
        logging.error(f"Error in evaluate_exam_with_complete_data: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({"status": "Failed", "error": str(e)}), 500



if __name__ == '__main__':
    key_file = os.environ.get('SERVICE_ACCOUNT_KEY_FILE')
    print(f"DEBUG: Using Key File: {key_file}")
    print("="*70)
    print("🚀 Starting Flask Paper Valuation Service")
    print("="*70)
    print("Available Endpoints:")
    print("  • POST /api/extract_answer_key_text - Extract answer key (real-time)")
    print("  • POST /api/save_answer_key - Save complete answer key")
    print("  • GET  /api/get_answer_key/<exam_id> - Retrieve answer key")
    print("  • GET  /api/list_answer_keys - List all answer keys")
    print("  • POST /api/seriesBundleEvaluate - Batch evaluation (with answer key support)")
    print("  • POST /api/evaluate_exam/<exam_id> - Evaluate ALL students in exam")  # ← MAIN ONE
    print("  • POST /api/evaluate_student/<exam_id>/<roll_no> - Evaluate single student")  # ← OPTIONAL
    print("="*70)
    print("Server running on http://localhost:5000")
    print("="*70)
    app.run(port=5000, debug=True, use_reloader=True)
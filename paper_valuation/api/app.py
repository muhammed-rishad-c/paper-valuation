import os
import sys
import traceback
import json
import uuid
from flask import Flask, request, jsonify

# --- Custom Imports ---
from paper_valuation.logging.logger import logging
from paper_valuation.api.utils import (
    evaluate_paper_individual, 
    evaluate_series_paper,
    extract_answer_key_text_util,
    save_answer_key_util,
    load_answer_keys,
    get_answer_key_by_id,
    get_exam_with_submissions
)

app = Flask(__name__)

# ============================================
# INDIVIDUAL VALUATION ROUTE
# ============================================

@app.route('/api/evaluate', methods=['POST'])
def evaluate_paper_endpoint():
    """Individual paper evaluation endpoint"""
    try:
        files = request.files.getlist('paper_images') 
        
        if not files or files[0].filename == '':
            logging.error("No images found in the request.")
            return jsonify({"status": "Failed", "error": "No images provided."}), 400
        
        logging.info("="*70)
        logging.info(f"Individual Valuation: Processing {len(files)} pages.")
        logging.info("="*70)
        
        results = evaluate_paper_individual(files)
        return results

    except Exception as e:
        error_message = f"Critical System Error: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_message)
        return jsonify({"status": "Failed", "error": str(e)}), 500

# ============================================
# ANSWER KEY MANAGEMENT ENDPOINTS
# ============================================

@app.route('/api/extract_answer_key_text', methods=['POST'])
def extract_answer_key_text():
    """Extract text from answer key image for teacher verification"""
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
    """Save complete answer key with metadata"""
    try:
        # Get JSON data
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
    """Retrieve a specific answer key by exam_id"""
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
    """List all available answer keys"""
    try:
        all_answer_keys = load_answer_keys()
        
        answer_key_list = []
        for key, value in all_answer_keys.items():
            # Handle both old and new structure
            if 'exam_metadata' in value:
                # New structure
                answer_key_list.append({
                    'exam_id': key,
                    'exam_name': value['exam_metadata'].get('exam_name', 'Unknown'),
                    'class': value['exam_metadata'].get('class', 'Unknown'),
                    'subject': value['exam_metadata'].get('subject', 'Unknown')
                })
            else:
                # Old structure (backward compatibility)
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

# ============================================
# SERIES BATCH EVALUATION ROUTE
# ============================================

@app.route('/api/seriesBundleEvaluate', methods=['POST'])
def evaluate_series_batch_handler():
    """Series batch evaluation endpoint with answer key support"""
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


# ============================================
# EXAM DATA WITH SUBMISSIONS ENDPOINT
# ============================================

@app.route('/api/get_exam_data/<exam_id>', methods=['GET'])
def get_complete_exam_data(exam_id):
    """Get complete exam data: teacher answers + all student submissions"""
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
# ============================================
# SERVER STARTUP
# ============================================

if __name__ == '__main__':
    key_file = os.environ.get('SERVICE_ACCOUNT_KEY_FILE')
    print(f"DEBUG: Using Key File: {key_file}")
    print("="*70)
    print("🚀 Starting Flask Paper Valuation Service")
    print("="*70)
    print("Available Endpoints:")
    print("  • POST /api/evaluate - Individual paper evaluation")
    print("  • POST /api/extract_answer_key_text - Extract answer key (real-time)")
    print("  • POST /api/save_answer_key - Save complete answer key")
    print("  • GET  /api/get_answer_key/<exam_id> - Retrieve answer key")
    print("  • GET  /api/list_answer_keys - List all answer keys")
    print("  • POST /api/seriesBundleEvaluate - Batch evaluation (with answer key support)")
    print("="*70)
    print("Server running on http://localhost:5000")
    print("="*70)
    app.run(port=5000, debug=True, use_reloader=True)
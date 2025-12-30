import os
import sys
import traceback
from flask import Flask, request, jsonify

# --- Custom Imports ---
from paper_valuation.logging.logger import logging
from paper_valuation.api.utils import evaluate_paper_individual, evaluate_series_paper

app = Flask(__name__)

# 1. Individual Valuation Route
@app.route('/api/evaluate', methods=['POST'])
def evaluate_paper_endpoint():
    try:
        files = request.files.getlist('paper_images') 
        
        if not files or files[0].filename == '':
            logging.error("No images found in the request.")
            return jsonify({"status": "Failed", "error": "No images provided."}), 400
        
        logging.info("="*70)
        logging.info(f"Individual Valuation: Processing {len(files)} pages.")
        logging.info("="*70)
        
        # Calls the individual helper in utils.py
        results = evaluate_paper_individual(files)
        return results

    except Exception as e:
        error_message = f"Critical System Error: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_message)
        return jsonify({"status": "Failed", "error": str(e)}), 500

@app.route('/api/seriesBundleEvaluate', methods=['POST'])
def evaluate_series_batch_handler():
    # Use a unique identifier or roll number for contextual logging
    manual_roll_no = request.form.get('manual_roll_no', 'N/A')
    
    logging.info(f"{'='*20} 🆕 NEW BATCH REQUEST {'='*20}")
    logging.info(f"Context: Student Roll No - {manual_roll_no}")

    try:
        # 1. Capture and Log Incoming Data Metadata
        id_file = request.files.get('identity_page')
        answer_files = request.files.getlist('paper_images')
        
        logging.info(f"Payload Details: ID Page: {id_file.filename if id_file else 'MISSING'}, "
                     f"Answer Pages Count: {len(answer_files)}")

        # 2. Basic Validation with Detailed Logging
        if not id_file:
            error_msg = f"Validation Error for Roll No {manual_roll_no}: Identity page (identity_page) is missing."
            logging.error(error_msg)
            return jsonify({"status": "Failed", "error": "Identity page is required."}), 400
        
        if not answer_files:
            logging.warning(f"No answer pages (paper_images) provided for Roll No {manual_roll_no}.")

        # 3. Processing with Step-by-Step Context
        logging.info(f"Starting paper evaluation logic for Student: {manual_roll_no}...")
        
        # Pass the manual_roll_no to the utility for fallback logging
        result = evaluate_series_paper(id_file, answer_files, manual_roll_no=manual_roll_no)
        
        logging.info(f"✅ Successfully processed evaluation for Roll No {manual_roll_no}.")
        return result
        
    except FileNotFoundError as fnf:
        logging.error(f"📁 File Error for Roll No {manual_roll_no}: {str(fnf)}")
        return jsonify({"status": "Failed", "error": "Internal file handling error."}), 500

    except Exception as e:
        # 4. Detailed Traceback for System Failures
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

if __name__ == '__main__':
    key_file = os.environ.get('SERVICE_ACCOUNT_KEY_FILE')
    print(f"DEBUG: Using Key File: {key_file}")
    print("Starting Flask Paper Valuation Service on port 5000...")
    app.run(port=5000, debug=True, use_reloader=True)
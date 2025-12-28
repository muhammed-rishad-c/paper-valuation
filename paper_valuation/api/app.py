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

# 2. Teacher Series Bundle Route
@app.route('/api/seriesBundleEvaluate', methods=['POST'])
def evaluate_series_batch_handler():
    """
    Renamed to evaluate_series_batch_handler to prevent 
    collision with the utility function.
    """
    try:
        # Match keys sent from Node.js controller
        id_file = request.files.get('identity_page')
        answer_files = request.files.getlist('paper_images')
        
        if not id_file:
            logging.error("Identity page missing in request.")
            return jsonify({"status": "Failed", "error": "Identity page is required."}), 400
        
        # Call the utility function from utils.py
        result = evaluate_series_paper(id_file, answer_files)
        return result
        
    except Exception as e:
        error_message = f"Critical System Error: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_message)
        return jsonify({"status": "Failed", "error": str(e)}), 500

if __name__ == '__main__':
    key_file = os.environ.get('SERVICE_ACCOUNT_KEY_FILE')
    print(f"DEBUG: Using Key File: {key_file}")
    print("Starting Flask Paper Valuation Service on port 5000...")
    app.run(port=5000, debug=True, use_reloader=False)
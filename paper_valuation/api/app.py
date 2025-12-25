import os
import sys
import tempfile
import traceback
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename


from paper_valuation.logging.logger import logging
from paper_valuation.api.utils import evaluate_paper_individual

app = Flask(__name__)



@app.route('/api/evaluate', methods=['POST'])
def evaluate_paper_endpoint():
    
    try:
        files = request.files.getlist('paper_images')   
        
        if not files or files[0].filename == '':
            logging.error("No images found in the request.")
            return jsonify({"status": "Failed", "error": "No images provided. Ensure the key is 'paper_images'."}), 400

        
        logging.info("="*70)
        logging.info("FILES RECEIVED BY FLASK (in order):")
        for index, file in enumerate(files):
            logging.info(f"  Page {index + 1}: {file.filename}")
        logging.info("="*70)
        
        results=evaluate_paper_individual(files)
        return results

    except Exception as e:
        error_message = f"Critical System Error: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_message)
        return jsonify({"status": "Failed", "error": str(e)}), 500



if __name__ == '__main__':

    key_file = os.environ.get('SERVICE_ACCOUNT_KEY_FILE')
    print(f"DEBUG: Using Key File: {key_file}")
    
    print("Starting Flask Paper Valuation Service on port 5000...")
    app.run(port=5000, debug=True, use_reloader=False)
import os
import sys
import tempfile
import traceback
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# --- Custom Imports ---
from paper_valuation.api.vision_segmentation import detect_and_segment_image
from paper_valuation.logging.logger import logging
# Ensure this utility function exists in your paper_valuation/api/utils.py
from paper_valuation.api.utils import merge_multi_page_result

app = Flask(__name__)

@app.route('/api/evaluate', methods=['POST'])
def evaluate_paper_endpoint():
    # 1. Retrieve all images uploaded under 'paper_images' (matching Node.js key)
    # Ensure this matches the Node.js key 'paper_images'
    files = request.files.getlist('paper_images')   
    
    if not files or files[0].filename == '':
        logging.error("No images found in the request.")
        return jsonify({"status": "Failed", "error": "No images provided. Ensure the key is 'paper_images'."}), 400

    all_pages_result = []

    try:
        # 2. Iterate through each page sequentially
        for index, file in enumerate(files):
            # Create a unique temp file for each page
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                file.save(tmp.name)
                temp_path = tmp.name
            
            logging.info(f"Processing Page {index + 1}: {temp_path}")
            
            # 3. Perform OCR and Segmentation on this page
            page_result = detect_and_segment_image(temp_path)
            all_pages_result.append(page_result)
            
            # 4. Clean up temp file immediately after processing the page
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logging.info(f"Cleaned up Page {index + 1} temp file.")

        # 5. Merge all individual page dictionaries into one complete script
        final_valuation = merge_multi_page_result(all_pages_result)
            
        return jsonify({
            "status": "Success",
            "recognition_result": final_valuation
        }), 200

    except Exception as e:
        error_message = f"Critical System Error: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_message)
        return jsonify({"status": "Failed", "error": str(e)}), 500

if __name__ == '__main__':
    # Sanity check for environment variables
    key_file = os.environ.get('SERVICE_ACCOUNT_KEY_FILE')
    print(f"DEBUG: Using Key File: {key_file}")
    
    print("Starting Flask Paper Valuation Service on port 5000...")
    app.run(port=5000, debug=True, use_reloader=False)
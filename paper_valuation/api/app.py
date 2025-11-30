
# ADD THIS TEMPORARILY:
import os
print(f"DEBUG: Credentials Path is: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")

from paper_valuation.api.vision_segmentation import detect_and_segment_image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os, sys
import tempfile 
import traceback 

from paper_valuation.logging.logger import logging
from paper_valuation.exception.custom_exception import CustomException 


app = Flask(__name__)


@app.route('/api/evaluate', methods=['POST'])
def evaluate_paper_endpoint():
    
    temp_path = None
    
    
    if 'paper_image' not in request.files:
        print("ERROR: Request missing 'paper_image' file.", file=sys.stderr)
        return jsonify({"status": "Failed", "error": "No image file provided. Check FormData key."}), 400

    file = request.files['paper_image']

    try:
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            file.save(tmp.name)
            temp_path = tmp.name
        
        
        print(f"INFO: Successfully received and saved file to {temp_path}", file=sys.stdout)
        
        recognition_result = detect_and_segment_image(temp_path)
        logging.info(f"API recognition successful. Sample text: {recognition_result.get('full_text_recognized', 'N/A')[:50]}...")
        
        
        
        
        return jsonify({
            "status": "API Recognition Test Success",
            "message": "Image processed by Google Vision API. Check 'recognition_result' for extracted text.",
            "recognition_result": recognition_result
        }), 200

    except Exception as e:
        
        error_message = f"Critical System Error during file receipt: {str(e)}\n{traceback.format_exc()}"
        print(error_message, file=sys.stderr)
        return jsonify({"status": "Failed", "error": "Internal Server Error during file receipt."}), 500
        
    finally:
        
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"INFO: Cleaned up temporary file: {temp_path}", file=sys.stdout)


if __name__ == '__main__':
    print("Starting Flask File Receipt Test Service on port 5000...", file=sys.stdout)
    app.run(port=5000, debug=True)
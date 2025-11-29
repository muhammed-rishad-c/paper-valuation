

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
        
        
        return jsonify({
            "status": "Success: File Received and Saved",
            "message": f"File successfully saved to temporary path: {temp_path}",
            "original_filename": secure_filename(file.filename)
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
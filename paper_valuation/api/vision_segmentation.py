# paper_valuation/api/vision_segmentation.py

from google.cloud import vision
import io
import google.auth # REQUIRED for loading credentials from file

# --- FINAL AUTHENTICATION FIX ---
# The clean, simple path to your key file
_SERVICE_ACCOUNT_KEY_FILE = r"E:\machine learning\project\paper-valuation\api key\key.json"

# ---------------------------------

def get_document_annotation(image_path: str):

    # 1. AUTHENTICATION: Load credentials directly from the file path
    # We pass the key file path directly to the client constructor
    credentials, project_id = google.auth.load_credentials_from_file(_SERVICE_ACCOUNT_KEY_FILE)

    client = vision.ImageAnnotatorClient(credentials=credentials)

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    # ... (rest of the function remains the same: image=vision.Image(content=content), etc.)

    image=vision.Image(content=content)
    image_context = vision.ImageContext(language_hints=["en-t-i0-handwrit", "en"])

    response=client.document_text_detection(
        image=image,
        image_context=image_context
    )

    return response.full_text_annotation

# ... (The rest of your detect_and_segment_image function remains the same)


def detect_and_segment_image(image_path:str)->dict:
    document_annotation=get_document_annotation(image_path)
    
    
    return {
        "full_text_recognized":document_annotation.text
    }
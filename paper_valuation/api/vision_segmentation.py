# paper_valuation/api/vision_segmentation.py (Full Code with FINAL Spacing Fix)

from google.cloud import vision
import google.auth
import io
import os
import re

from wordsegment import load, segment
load()

from dotenv import load_dotenv
load_dotenv()
# --- Authentication Configuration ---
_SERVICE_ACCOUNT_KEY_FILE = os.environ.get("SERVICE_ACCOUNT_KEY_FILE")

if not _SERVICE_ACCOUNT_KEY_FILE:
     raise ValueError("SERVICE_ACCOUNT_KEY_FILE not found. Check if .env is loaded correctly in app.py and variable name is correct.")

# --- Helper Functions for Segmentation ---

def is_question_label(text: str) -> bool:
    # Pattern: Optional starting letter/symbol (@, Q), followed by one or more digits, 
    # followed by optional punctuation (: or .).
    return re.match(r'^[A-Z@]?\s*\d+\s*[:\.]?', text.strip(), re.IGNORECASE) is not None

def extract_block_data(document_annotation) -> list:
    """Extracts all text blocks (paragraphs) with their bounding box (X and Y coordinates)."""
    block_data = []
    
    for page in document_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                # Get the full text of the paragraph
                text = "".join([symbol.text for word in paragraph.words for symbol in word.symbols])
                
                # Get the top-left coordinate (normalized to 1000)
                bbox = [(v.x, v.y) for v in paragraph.bounding_box.vertices]
                min_x = min(v[0] for v in bbox)
                min_y = min(v[1] for v in bbox)
                
                is_label_block = is_question_label(text.split()[0] if text else "")
                
                block_data.append({
                    'text': text,
                    'x': min_x,
                    'y': min_y,
                    'is_label': is_label_block 
                })
    return block_data

def reconstruct_answer_text(block_data_list):
    """
    FINAL FIX: Uses the wordsegment library to intelligently split the mashed text 
    into readable English words.
    """
    # 1. Combine all blocks into one giant congested string
    full_congested_text = "".join(block['text'] for block in block_data_list)
    
    # 2. Use wordsegment to split the congested text into a list of valid words
    # Example: "theanswerofgivenanswer" -> ["the", "answer", "of", "given", "answer"]
    segmented_words = segment(full_congested_text)
    
    # 3. Join the words with spaces for the final clean answer
    return ' '.join(segmented_words)

# --- Main API Caller ---

def get_document_annotation(image_path: str):
    """Initializes the Vision client and requests full text detection."""
    credentials, project_id = google.auth.load_credentials_from_file(_SERVICE_ACCOUNT_KEY_FILE)
    client = vision.ImageAnnotatorClient(credentials=credentials)
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    image_context = vision.ImageContext(language_hints=["en-t-i0-handwrit", "en"])
    response = client.document_text_detection(
        image=image,
        image_context=image_context
    )
    return response.full_text_annotation

# --- Main Segmentation Logic (Segment_answers remains unchanged from previous step) ---

def segment_answers(document_annotation) -> dict:
    """
    Segments the document into labeled answers based on the vertical split strategy.
    (This function body is copied from the last fully working version, only 
     the supporting functions and reconstruct_answer_text are updated).
    """
    SPLIT_X = 350
    block_data = extract_block_data(document_annotation)
    
    # --- DEBUGGING OUTPUT (For Console Inspection) ---
    print("\n--- ALL EXTRACTED BLOCKS (X, Y, is_label, Text) ---")
    for block in block_data:
        if block['x'] < 500:
            print(f"X:{block['x']:<4} | Y:{block['y']:<4} | Label:{block['is_label']:<5} | Text: {block['text'][:50]}")
    print("----------------------------------------------------\n")
    
    # 2. Identify Question Boundaries (Labels on the LEFT side)
    question_boundaries = []
    
    for block in block_data:
        if block['is_label'] and block['x'] < SPLIT_X:
            match = re.search(r'(\d+)', block['text'].split()[0])
            if match:
                clean_label = 'Q' + match.group(1).strip()
            else:
                continue 
            question_boundaries.append({
                'label': clean_label,
                'y_start': block['y']
            })
            
    question_boundaries.sort(key=lambda x: x['y_start'])
    
    # --- Q1 FORGIVENESS FIX ---
    if not any(qb['label'].upper() == 'Q1' for qb in question_boundaries) and block_data:
        first_block_on_left = min((b for b in block_data if b['x'] < SPLIT_X), 
                                  key=lambda x: x['y'], default=None)
        if first_block_on_left and (not question_boundaries or first_block_on_left['y'] < question_boundaries[0]['y_start']):
            question_boundaries.insert(0, {
                'label': 'Q1',
                'y_start': first_block_on_left['y']
            })
    # ----------------------------

    # --- DEBUGGING OUTPUT ---
    print("\n--- IDENTIFIED QUESTION BOUNDARIES ---")
    for qb in question_boundaries:
        print(f"Label: {qb['label']:<5} | Y-Start: {qb['y_start']}")
    print("--------------------------------------\n")

    # 3. Group Answer Text based on Vertical Bands
    segmented_answers = {}
    
    for i, q_boundary in enumerate(question_boundaries):
        
        y_start_boundary = q_boundary['y_start']
        y_end_boundary = question_boundaries[i + 1]['y_start'] if i + 1 < len(question_boundaries) else float('inf')

        answer_blocks = []
        
        for block in block_data:
            if y_start_boundary <= block['y'] < y_end_boundary:
                
                if block['is_label'] and block['x'] < SPLIT_X and block['y'] == y_start_boundary:
                    block_copy = block.copy()
                    parts = block_copy['text'].split(':', 1)
                    if len(parts) > 1:
                        block_copy['text'] = parts[1].strip()
                    elif len(block_copy['text'].split()) > 1:
                        block_copy['text'] = ' '.join(block_copy['text'].split()[1:]).strip()
                    else:
                        continue 
                        
                    answer_blocks.append(block_copy)
                
                else:
                    answer_blocks.append(block)
        
        label_key = q_boundary['label']
        # CALL THE RECONSTRUCTOR HERE:
        segmented_answers[label_key] = reconstruct_answer_text(answer_blocks)
    
    print("\n--- FINAL SEGMENTATION OUTPUT ---")
    print(segmented_answers)
    print("---------------------------------\n")

    return segmented_answers

# --- Main Wrapper Function ---

def detect_and_segment_image(image_path:str)->dict:
    """Runs OCR and then segments the result for valuation."""
    
    document_annotation = get_document_annotation(image_path)
    segmented_data = segment_answers(document_annotation)
    
    return segmented_data
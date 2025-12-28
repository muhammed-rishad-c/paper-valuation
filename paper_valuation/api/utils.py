import re
from flask import jsonify,request,Flask
import tempfile,os,sys
from paper_valuation.logging.logger import logging
from paper_valuation.api.vision_segmentation import detect_and_segment_image,get_document_annotation


def merge_multi_page_result(all_pages_list):
    
    merged_answers = {}
    last_q_label = None
    
    for page_index, page in enumerate(all_pages_list):
        answers = page.get('answers', {})
        
        if 'UNLABELED_CONTINUATION' in answers:
            unlabeled_text = answers.pop('UNLABELED_CONTINUATION')
            
            if last_q_label and last_q_label in merged_answers:
                merged_answers[last_q_label] += " " + unlabeled_text.strip()
                print(f"⚠️  Page {page_index + 1}: Unlabeled continuation appended to {last_q_label}")
                print(f"    (Student should have written '{last_q_label}' at the top of this page)")
            else:
                if 'Q1' in merged_answers:
                    merged_answers['Q1'] = unlabeled_text.strip() + " " + merged_answers['Q1']
                else:
                    merged_answers['Q1'] = unlabeled_text.strip()
                last_q_label = 'Q1'
                print(f"⚠️  Page {page_index + 1}: Unlabeled page assigned to Q1 by default")

        for q_label, text in answers.items():
            if q_label in merged_answers:
                merged_answers[q_label] += " " + text.strip()
                print(f"✅ Page {page_index + 1}: {q_label} continuation merged (student correctly labeled it)")
            else:
                merged_answers[q_label] = text.strip()
                print(f"✅ Page {page_index + 1}: Added {q_label}")
            
            last_q_label = q_label
    
    sorted_answers = dict(sorted(
        merged_answers.items(),
        key=lambda x: int(re.search(r'\d+', x[0]).group()) if re.search(r'\d+', x[0]) else 0
    ))
    
    print(f"\n📊 Final Merge Summary: {len(sorted_answers)} questions across {len(all_pages_list)} pages")
    
    return {"answers": sorted_answers, "total_pages": len(all_pages_list)}

def evaluate_paper_individual(files):
    try:
        all_page_result=[]
        for index,file in enumerate(files):
            with tempfile.NamedTemporaryFile(delete=False,suffix='.jpg') as tmp:
                file.save(tmp.name)
                temp_path=tmp.name
                
            logging.info(f"Processing Page {index + 1}: {file.filename} -> {temp_path}")
            page_result=detect_and_segment_image(temp_path,debug=True)
            all_page_result.append(page_result)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logging.info(f"Cleaned up Page {index + 1} temp file.")
                
                
        final_valuation=merge_multi_page_result(all_page_result)
        return jsonify({
            "status": "Success",
            "recognition_result": final_valuation
        }), 200

    except Exception as e:
        logging.error(e)
        return jsonify({"status": "Failed", "error": str(e)}), 500
    
def extract_series_identity(document_annotation):
    
    full_text = document_annotation.text
    details = {
        "name": "Unknown",
        "class": "Unknown",
        "subject": "Unknown",
        "roll_no": "Unknown"
    }

    patterns = {
        "name": r"Name\s*[:\-]\s*([A-Za-z\s]+)",
        "class": r"Class\s*[:\-]\s*([A-Za-z0-9\s]+)",
        "subject": r"Subject\s*[:\-]\s*([A-Za-z\s]+)",
        "roll_no": r"Roll\s*(?:No|#)?\s*[:\-]\s*(\d+)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            # Clean the extracted value
            details[key] = match.group(1).strip().split('\n')[0]

    return details

def evaluate_series_paper(student_id,answer_files):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            student_id.save(tmp.name)
            id_temp_path = tmp.name
            
        logging.info(f"Extracting identity from: {student_id.filename}")
        id_annotation = get_document_annotation(id_temp_path)
        student_info = extract_series_identity(id_annotation)
        
        if os.path.exists(id_temp_path):
            os.remove(id_temp_path)
            
        all_pages_result = []
        for index, file in enumerate(answer_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                file.save(tmp.name)
                temp_path = tmp.name
            
            logging.info(f"Processing Answer Page {index + 1} for {student_info['name']}")
            
            # Perform OCR and Segmentation
            page_result = detect_and_segment_image(temp_path, debug=True)
            all_pages_result.append(page_result)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # 4. Merge all answers into one structured response
        final_valuation = merge_multi_page_result(all_pages_result)
            
        return jsonify({
            "status": "Success",
            "student_info": student_info,
            "recognition_result": final_valuation
        }), 200
        
        
    except Exception as e:
        logging.error(e)
        return jsonify({"status": "Failed", "error": str(e)}), 500
        
            
            
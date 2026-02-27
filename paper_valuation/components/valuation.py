from sentence_transformers import SentenceTransformer, util
import re
import sys
from paper_valuation.exception.custom_exception import CustomException
from paper_valuation.logging.logger import logging

model = SentenceTransformer('all-MiniLM-L6-v2')

# ============================================
# NEW: ANSWER CLEANING FUNCTION
# ============================================
def clean_teacher_answer(answer: str) -> str:
    """
    Remove grading instructions and notes from teacher answer
    to get only the actual answer content
    """
    try:
        # Remove grading instruction patterns
        patterns = [
            r'To earn full marks.*?(?=\.|$)',
            r'Grading Note:.*?(?=\.|$)',
            r'The student must.*?(?=\.|$)',
            r'Marks are awarded.*?(?=\.|$)',
            r'\([0-9]+\s*Marks?\):.*?(?=\.\.|\n|$)',
            r'Each.*?should.*?(?=\.|$)',
            r'Focus on.*?(?=\.|$)',
            r'Implementation of.*?(?=\.|$)'
        ]
        
        cleaned = answer
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove extra whitespace and newlines
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    except Exception as e:
        logging.warning(f"Error cleaning teacher answer: {e}")
        return answer  # Return original if cleaning fails


def normalize_text(text: str) -> str:
    """
    Normalize text for better comparison
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def round_by_half(score: float) -> float:
    return round(score * 2) / 2


# ============================================
# IMPROVED: More Lenient Marking Scheme
# ============================================
def calculate_marks(similarity_score: float, max_mark: float, threshold: float = 0.45, exponent: float = 0.5) -> float:
    """
    Convert similarity score to marks with more lenient grading
    
    Changes:
    - Lowered default threshold from 0.55 to 0.45
    - Lowered exponent from 0.6 to 0.5 for more linear scoring
    """
    try:
        logging.info(f"Calculating marks: similarity={similarity_score:.3f}, max={max_mark}, threshold={threshold}")
        
        if similarity_score < threshold:
            logging.info(f"Below threshold ({threshold}), returning 0")
            return 0
        
        # Scale from threshold to 1.0
        scale = (similarity_score - threshold) / (1.0 - threshold)
        
        # Apply curve (lower exponent = more lenient)
        curved_scale = scale ** exponent
        
        # Calculate mark
        mark = curved_scale * max_mark
        
        # Round to nearest 0.5
        final_mark = round_by_half(round(mark, 2))
        
        logging.info(f"Final mark: {final_mark}/{max_mark}")
        return final_mark
        
    except Exception as e:
        raise CustomException(e, sys)


def short_answer_valuation(teacher_answer: str, student_answer: str) -> float:
    """
    Calculate semantic similarity between teacher and student answers
    """
    try:
        logging.info("Computing semantic similarity...")
        
        student_answer_embedded = model.encode(student_answer, convert_to_tensor=True)
        teacher_answer_embedded = model.encode(teacher_answer, convert_to_tensor=True)
        
        similarity = util.cos_sim(student_answer_embedded, teacher_answer_embedded)
        
        similarity_score = similarity.item()
        logging.info(f"Similarity score: {similarity_score:.3f}")
        
        return similarity_score
        
    except Exception as e:
        raise CustomException(e, sys)


def smart_paragraph_split(text: str) -> list:
    """
    Split long answers into logical paragraphs for point-by-point comparison
    """
    try:
        logging.info("Splitting text into paragraphs...")
        text = text.strip()
        
        # First try double newline split
        paragraph = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # If too few paragraphs, try single newline
        if len(paragraph) <= 2:
            single_line_paragraph = [p.strip() for p in text.split('\n') if p.strip()]
            if len(single_line_paragraph) > len(paragraph):
                paragraph = single_line_paragraph
        
        # If still too few, split by sentences
        if len(paragraph) <= 2:
            sentence = re.split(r'[.!?]+', text)
            sentence = [s.strip() for s in sentence if s.strip() and len(s) > 10]
            
            if len(sentence) > 3:
                chunk_size = max(1, min(3, len(sentence) // 3))
                paragraph = []
                
                for i in range(0, len(sentence), chunk_size):
                    chunk_sentence = sentence[i:i + chunk_size]
                    chunk = '. '.join(chunk_sentence)
                    if not chunk.endswith('.'):
                        chunk += '.'
                    paragraph.append(chunk)
        
        # Split very long paragraphs
        final_paragraph = []
        for para in paragraph:
            if len(para) > 300:
                sentences = re.split(r'[.!?]+', para)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                if len(sentences) > 2:
                    for i in range(0, len(sentences), 2):
                        chunk_sentences = sentences[i:i + 2]
                        chunk = '. '.join(chunk_sentences)
                        if not chunk.endswith('.'):
                            chunk += '.'
                        final_paragraph.append(chunk)
                else:
                    final_paragraph.append(para)
            else:
                final_paragraph.append(para)
        
        logging.info(f"Split into {len(final_paragraph)} paragraphs")
        return final_paragraph
        
    except Exception as e:
        raise CustomException(e, sys)


# ============================================
# IMPROVED: More Lenient Point-by-Point
# ============================================
def point_by_point_valuation(teacher_key_point: list, student_answer: str, mark_per_point: float, threshold: float = 0.35) -> dict:
    """
    Evaluate student answer against each teacher key point
    
    Changes:
    - Lowered threshold from 0.4 to 0.35
    """
    try:
        logging.info(f"Point-by-point evaluation started for {len(teacher_key_point)} key points")
        
        student_paragraph = smart_paragraph_split(student_answer)
        total_mark = len(teacher_key_point) * mark_per_point
        total_mark_obtained = 0.0
        detailed_result = []
        
        for i, key_point in enumerate(teacher_key_point):
            logging.info(f"Evaluating key point {i+1}: {key_point[:50]}...")
            
            key_point_embedded = model.encode(key_point, convert_to_tensor=True)
            best_score_key_point = 0.0
            best_matching_paragraph = ""
            
            for paragraph in student_paragraph:
                student_paragraph_embedded = model.encode(paragraph, convert_to_tensor=True)
                score = util.cos_sim(key_point_embedded, student_paragraph_embedded)
                
                if score.item() > best_score_key_point:
                    best_score_key_point = score.item()
                    best_matching_paragraph = paragraph[:50] + "...." if len(paragraph) > 50 else paragraph
            
            mark_of_point = calculate_marks(best_score_key_point, mark_per_point, threshold)
            total_mark_obtained += mark_of_point
            
            logging.info(f"Key point {i+1} score: {mark_of_point}/{mark_per_point} (similarity: {best_score_key_point:.3f})")
            
            detailed_result.append({
                "key_point": key_point,
                "mark_scored": mark_of_point,
                "max_mark_of_point": mark_per_point,
                "similarity_score": best_score_key_point,
                "best_match": best_matching_paragraph
            })
        
        logging.info(f"Point-by-point total: {total_mark_obtained}/{total_mark}")
        
        return {
            "details": detailed_result,
            "total_mark_scored": total_mark_obtained,
            "total_mark": total_mark
        }
        
    except Exception as e:
        raise CustomException(e, sys)


# ============================================
# IMPROVED: More Lenient Holistic
# ============================================
def holistic_valuation(student_answer: str, teacher_key_point: list, total_question_mark: int, threshold: float = 0.30) -> dict:
    """
    Evaluate student answer holistically against all key points combined
    
    Changes:
    - Lowered threshold from 0.35 to 0.30
    """
    try:
        logging.info("Holistic evaluation started")
        
        teacher_answer = ' '.join(teacher_key_point)
        
        student_answer_embedded = model.encode(student_answer, convert_to_tensor=True)
        teacher_answer_embedded = model.encode(teacher_answer, convert_to_tensor=True)
        
        similarity_score = util.cos_sim(student_answer_embedded, teacher_answer_embedded)
        
        mark = calculate_marks(similarity_score.item(), total_question_mark, threshold)
        
        logging.info(f"Holistic score: {mark}/{total_question_mark} (similarity: {similarity_score.item():.3f})")
        
        return {
            "total_mark_scored": mark,
            "total_mark": total_question_mark,
            "similarity": similarity_score
        }
        
    except Exception as e:
        raise CustomException(e, sys)


def advanced_long_valuation(teacher_key_point: list, student_answer: str, total_question_mark: int, 
                           holistic_threshold: float = 0.30, point_threshold: float = 0.35) -> dict:
    """
    Evaluate long answer using BOTH holistic and point-by-point, take the better score
    
    Changes:
    - Lowered holistic_threshold from 0.35 to 0.30
    - Lowered point_threshold from 0.4 to 0.35
    """
    try:
        logging.info("=" * 60)
        logging.info("Advanced long answer evaluation started")
        logging.info(f"Teacher has {len(teacher_key_point)} key points")
        logging.info(f"Total marks: {total_question_mark}")
        
        # Clean teacher key points
        cleaned_key_points = [clean_teacher_answer(kp) for kp in teacher_key_point]
        
        # Holistic evaluation
        holistic_result = holistic_valuation(student_answer, cleaned_key_points, total_question_mark, holistic_threshold)
        
        # Point-by-point evaluation
        mark_per_point = total_question_mark / len(cleaned_key_points)
        point_by_point_result = point_by_point_valuation(cleaned_key_points, student_answer, mark_per_point, point_threshold)
        
        # Take the better score
        final_score = max(holistic_result["total_mark_scored"], point_by_point_result['total_mark_scored'])
        
        logging.info(f"Holistic score: {holistic_result['total_mark_scored']}")
        logging.info(f"Point-by-point score: {point_by_point_result['total_mark_scored']}")
        logging.info(f"Final score (max): {final_score}/{total_question_mark}")
        logging.info("=" * 60)
        
        return {
            "final_score": final_score,
            "holistc_result": holistic_result,
            "point_by_point_result": point_by_point_result
        }
        
    except Exception as e:
        raise CustomException(e, sys)


# ============================================
# IMPROVED: Short Answer Evaluation
# ============================================
def evaluation_short_answer(student_answer: str, teacher_answer: str, max_mark: int, threshold: float = 0.45) -> float:
    """
    Evaluate short answer
    
    Changes:
    - Added answer cleaning
    - Lowered threshold from 0.55 to 0.45
    - Added detailed logging
    """
    try:
        logging.info("=" * 60)
        logging.info("Short answer evaluation started")
        logging.info(f"Max marks: {max_mark}")
        
        # Clean teacher answer
        teacher_answer_cleaned = clean_teacher_answer(teacher_answer)
        
        logging.info(f"Teacher answer (cleaned): {teacher_answer_cleaned[:100]}...")
        logging.info(f"Student answer: {student_answer[:100]}...")
        
        # Calculate similarity
        similarity = short_answer_valuation(teacher_answer_cleaned, student_answer)
        
        # Convert to marks
        final_mark = calculate_marks(similarity, max_mark, threshold)
        
        logging.info(f"Short answer result: {final_mark}/{max_mark}")
        logging.info("=" * 60)
        
        return final_mark
        
    except Exception as e:
        raise CustomException(e, sys)


# ============================================
# IMPROVED: Long Answer Evaluation
# ============================================
def evaluation_long_answer(student_answer: str, teacher_answer: list, max_mark: int, 
                          holistic_threshold: float = 0.30, point_threshold: float = 0.35) -> float:
    """
    Evaluate long answer
    
    Changes:
    - Lowered holistic_threshold from 0.50 to 0.30
    - Lowered point_threshold from 0.55 to 0.35
    - Added answer cleaning
    """
    try:
        # Clean teacher key points
        cleaned_teacher_answer = [clean_teacher_answer(point) for point in teacher_answer]
        
        report = advanced_long_valuation(cleaned_teacher_answer, student_answer, max_mark, holistic_threshold, point_threshold)
        
        return report["final_score"]
        
    except Exception as e:
        raise CustomException(e, sys)
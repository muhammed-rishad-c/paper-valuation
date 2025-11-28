from sentence_transformers import SentenceTransformer,util
import re,sys
from paper_valuation.exception.custom_exception import CustomException
from paper_valuation.logging.logger import logging


model=SentenceTransformer('all-MiniLM-L6-v2')

def round_by_half(score:float)->float:
    return round(score*2)/2

def calculate_marks(similarity_score:float,max_mark:float,threshold:float=0.55,exponent:float=0.6)->float:
    try:
        logging.info("calculating mark is started")
        if similarity_score < threshold:
            return 0
        
        scale = (similarity_score - threshold) / (1.0 - threshold)
        curved_scale = scale ** exponent
        mark = curved_scale * max_mark
        final_mark = round_by_half(round(mark, 2))
        logging.info("mark is returned")
        return final_mark
    except Exception as e:
        raise CustomException(e, sys)





def short_answer_valuation(teacher_answer:str,student_answer:str)->float:
    try:
        logging.info("short answer valuation is done")
        student_answer_embedded=model.encode(student_answer,convert_to_tensor=True)
        teacher_answer_embedded=model.encode(teacher_answer,convert_to_tensor=True)
        similarity=util.cos_sim(student_answer_embedded,teacher_answer_embedded)
        logging.info("similarity of short answer is returned")
        return similarity.item()
    except Exception as e:
        raise CustomException(e,sys)


def smart_paragraph_split(text:str)->list:
    try:
        logging.info("splitting paragraph to chunks is started")
        text=text.strip()
        
        paragraph=[p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraph)<=2:
            single_line_paragraph=[p.strip() for p in text.split('\n') if p.strip()]
            if len(single_line_paragraph)>len(paragraph):
                paragraph=single_line_paragraph
                
        if len(paragraph)<=2:
            sentence=re.split(r'[.!?]+',text)
            sentence=[s.strip() for s in sentence if s.strip() and len(s)>10]
            
            if len(sentence)>3:
                chunk_size=max(1,min(3,len(sentence)//3))
                paragraph=[]
                
                for i in range(0,len(sentence),chunk_size):
                    chunk_sentence=sentence[i:i+chunk_size]
                    chunk='.'.join(chunk_sentence)
                    if not chunk.endswith('.'):
                        chunk+='.'
                    paragraph.append(chunk)
                    
        final_paragraph=[]
        for para in paragraph:
            
            if len(para) > 300:
                sentences = re.split(r'[.!?]+', para)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                if len(sentences) > 2:
                    # Split long paragraph into 2-sentence chunks
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
        logging.info("paragraph splitting is done")
        return final_paragraph
    except Exception as e:
        raise CustomException(e,sys)
    


def point_by_point_valuation(teacher_key_point:list,student_answer:str,mark_per_point:float,threshold:float=0.4)->dict:
    try:
        logging.info("point by point by valuation is started")
        student_paragraph=smart_paragraph_split(student_answer)
        total_mark=len(teacher_key_point)*mark_per_point
        total_mark_obtained=0.0
        detailed_result=[]
        
        for i,key_point in enumerate(teacher_key_point):
            key_point_embedded=model.encode(key_point,convert_to_tensor=True)
            best_score_key_point=0.0
            best_matching_paragraph=""
            
            for paragraph in student_paragraph:
                student_paragraph_embedded=model.encode(paragraph,convert_to_tensor=True)
                score=util.cos_sim(key_point_embedded,student_paragraph_embedded)
                if score.item()>best_score_key_point:
                    best_score_key_point=score.item()
                    best_matching_paragraph=paragraph[:50]+"...."if len(paragraph)>50 else paragraph
                    
            mark_of_point=calculate_marks(best_score_key_point,mark_per_point,threshold)
            total_mark_obtained+=mark_of_point
            
            detailed_result.append({
                "key_point":key_point,
                "mark_scored":mark_of_point,
                "max_mark_of_point":mark_per_point,
                "similarity_score":best_score_key_point,
                "best_match":best_matching_paragraph
            })
        logging.info("point evaluation is done")
        
        return {
            "details":detailed_result,
            "total_mark_scored":total_mark_obtained,
            "total_mark":total_mark
        }
    except Exception as e:
        raise CustomException(e,sys)
    
def holistic_valuation(student_answer:str,teacher_key_point:list,total_question_mark:int,threshold:float=0.35)->dict:
    try:
        logging.info("holisitc evaluation is started")
        teacher_answer=' '.join(teacher_key_point)
        student_answer_embedded=model.encode(student_answer,convert_to_tensor=True)
        teacher_answer_embedded=model.encode(teacher_answer,convert_to_tensor=True)
        similarity_score=util.cos_sim(student_answer_embedded,teacher_answer_embedded)
        mark=calculate_marks(similarity_score.item(),total_question_mark,threshold)
        final_mark=mark
        logging.info("holistic evaluation is done")
        return {
            "total_mark_scored":final_mark,
            "total_mark":total_question_mark,
            "similarity":similarity_score
        }
    except Exception as e:
        raise CustomException(e,sys)
    
def advanced_long_valuation(teacher_key_point:list,student_answer:str,total_question_mark:int,holistic_threshold:float=0.35,point_threshold:float=0.4)->dict:
    try:
        logging.info("data are prepared to sent on holistic evalution")
        
        holistic_result=holistic_valuation(student_answer,teacher_key_point,total_question_mark,holistic_threshold)
        mark_per_point=total_question_mark/len(teacher_key_point)
        
        logging.info("now data are prepared to sent point evaluation")
        point_by_point_result=point_by_point_valuation(teacher_key_point,student_answer,mark_per_point,point_threshold)
        
        final_score=max(holistic_result["total_mark_scored"],point_by_point_result['total_mark_scored'])
        logging.info("both methods of evaluation is done")
        return {
            "final_score":final_score,
            "holistc_result":holistic_result,
            "point_by_point_result":point_by_point_result
        }
    except Exception as e:
        raise CustomException(e,sys)
        
def evaluation_short_answer(student_answer:str,teacher_answer:str,max_mark:int,threshold:float=0.55)->float:
    try:
        logging.info("short answer valution is started")
        similarity=short_answer_valuation(student_answer,teacher_answer)
        final_mark=calculate_marks(similarity,max_mark,threshold)
        logging.info("short answer valuation is done mark is obtained")
        return final_mark
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluation_long_answer(student_answer:str,teacher_answer:list,max_mark:int,holistic_threshold:float=0.50,point_threshold:float=0.55)->float:
    try:
        # print(f"studnent answer is this {student_answer}")
        # print(f"\n\nteacher key point is {teacher_answer}")
        report=advanced_long_valuation(teacher_answer,student_answer,max_mark,holistic_threshold,point_threshold)
        return report["final_score"]
    except Exception as e:
        raise CustomException(e,sys)
        

        
    
from sentence_transformers import SentenceTransformer,util
import re

model=SentenceTransformer('all-MiniLM-L6-v2')

def round_by_half(score:float)->float:
    return round(score*2)/2

def calculate_marks(similarity_score:float,max_mark:float,threshold:float=0.55,exponent:float=0.6)->float:
    
    if similarity_score<threshold:
        return 0
    
    scale=(similarity_score-threshold)/(1.0-threshold)
    curved_scale=scale**exponent
    mark=curved_scale*max_mark
    final_mark=round_by_half(round(mark,2))
    return final_mark





def short_answer_valuation(teacher_answer:str,student_answer:str)->float:
    student_answer_embedded=model.encode(student_answer,convert_to_tensor=True)
    teacher_answer_embedded=model.encode(teacher_answer,convert_to_tensor=True)
    similarity=util.cos_sim(student_answer_embedded,teacher_answer_embedded)
    return similarity.item()


def smart_paragraph_split(text:str)->list:
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
            sentence = re.split(r'[.!?]+', para)
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

    return final_paragraph


def point_by_point_valuation(teacher_key_point:list,student_answer:str,mark_per_point:float,threshold:float=0.4)->dict:
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
    
    return {
        "details":detailed_result,
        "total_mark_scored":total_mark_obtained,
        "total_mark":total_mark
    }
    
def holistic_valuation(student_answer:str,teacher_key_point:list,total_question_mark:int,threshold:float=0.35)->dict:
    teacher_answer=' '.join(teacher_key_point)
    student_answer_embedded=model.encode(student_answer,convert_to_tensor=True)
    teacher_answer_embedded=model.encode(teacher_answer,convert_to_tensor=True)
    similarity_score=util.cos_sim(student_answer_embedded,teacher_answer_embedded)
    mark=calculate_marks(similarity_score.item(),total_question_mark,threshold)
    final_mark=mark
    
    return {
        "total_mark_scored":final_mark,
        "total_mark":total_question_mark,
        "similarity":similarity_score
    }
    
def advanced_long_valuation(teacher_key_point:list,student_answer:str,total_question_mark:int,holistic_threshold:float=0.35,point_threshold:float=0.4)->dict:
        holistic_result=holistic_valuation(student_answer,teacher_key_point,total_question_mark,holistic_threshold)
        mark_per_point=total_question_mark/len(teacher_key_point)
        point_by_point_result=point_by_point_valuation(teacher_key_point,student_answer,mark_per_point,point_threshold)
        
        final_score=max(holistic_result["total_mark_scored"],point_by_point_result['total_mark_scored'])
        
        return {
            "final_score":final_score,
            "holistc_result":holistic_result,
            "point_by_point_result":point_by_point_result
        }
        
def evaluation_short_answer(student_answer:str,teacher_answer:str,max_mark:int,threshold:float=0.55)->float:
    similarity=short_answer_valuation(student_answer,teacher_answer)
    final_mark=calculate_marks(similarity,max_mark,threshold)
    return final_mark
        
        
if __name__=="__main__":
    q1_student_correct = "Mitochondria provides the energy for cellular functions."
    q1_student_incorrect = "The nucleus controls the cell's activities."
    q1_teacher = "The mitochondria is the powerhouse of the cell."
    
    print("\n short answer valuation :::: \n")
    
    similarity_q1=short_answer_valuation(q1_teacher,q1_student_correct)
    mark_q1=calculate_marks(similarity_q1,3,0.55)
    
    print(f"mark of student one is {mark_q1}")
    
    similarity_q2=short_answer_valuation(q1_teacher,q1_student_incorrect)
    mark_q2=calculate_marks(similarity_q2,3,0.55)
    
    print(f"mark of student one is {mark_q2}")
    
    
    print("short answer valuation is done")
    
    
    teacher_points = [
    "Water turns into vapor when heated by the sun from rivers, lakes, and oceans.",
    "Water vapor rises and cools in the atmosphere, forming clouds through condensation.",
    "Water falls back to Earth from clouds as rain, snow, or other precipitation.",
    "Water collects in rivers, lakes, and oceans, completing the water cycle."
    ]
    
    student_answers_to_test = [
        # 1. The "Technical" Answer (Correct and uses advanced terms)
        """
        The water cycle is driven by solar energy. It begins with evaporation, where water from oceans and lakes turns into vapor and rises. In the colder upper atmosphere, condensation occurs, transforming the vapor into liquid droplets to form clouds. When these droplets grow heavy, they fall as precipitation, such as rain or snow. Finally, the collection phase gathers this water in rivers and oceans, completing the hydrologic cycle.
        """,

        # 2. The "Partially Correct" Answer (Misses a key concept)
        """
        The sun shines on the ocean and the water disappears up into the air. It floats up high and makes clouds. After a while, the rain falls down from the clouds and lands in the rivers, which go back to the ocean.
        """,

        # 3. The "Vague and Poorly Structured" Answer (Lacks detail)
        """
        Water goes up and water comes down. The sun makes it go up from the big lakes. Then it becomes clouds. The clouds move around and then water comes down. So the water is always moving around in a big circle from the ground to the sky and back.
        """,

        # 4. The "Poorly Formatted" Answer (Good content, no paragraphs)
        """
        The first step is the sun heating water from places like rivers and oceans so it turns into a gas called water vapor and rises high into the atmosphere. As the vapor gets higher and the air gets colder, it turns back into tiny water droplets, and this is how clouds are formed through a process called condensation. Eventually the droplets in the clouds get too heavy and fall back to the ground as precipitation, which can be in the form of rain or snow. That water then collects in rivers and flows back to the ocean and the whole water cycle can start over.
        """,

        # 5. The "Incorrect / Off-Topic" Answer (Should score zero)
        """
        The sun is a star that gives us warmth. The ocean is big and salty and has many fish. Clouds in the sky can block the sun, and sometimes rain comes from them. Weather is very interesting.
        """
    ]
    
    for i,student_answer in enumerate(student_answers_to_test):
        report=advanced_long_valuation(teacher_points,student_answer,10)
        
        print(f"{i+1} student scores mark of {report['final_score']} / 10 ")
        
    
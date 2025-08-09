from paper_valuation.exception.custom_exception import CustomException
from paper_valuation.logging.logger import logging
from paper_valuation.components.constant.valuation_data import student_answers_for_testing,teacher_answer_key_3marks
from paper_valuation.components.valuation import evaluation_short_answer
import pandas as pd
import sys

flat_data=[]
try:
    logging.info("taking data from source is initialized")
    
    for student in student_answers_for_testing:
        for q_id,answer in student['answers'].items():
            flat_data.append({
                "student_id":student['student_id'],
                "question_no":q_id,
                "student_answer":answer
            })
            
    df=pd.DataFrame(flat_data)

    teacher_answer={q_id:data['answer'] for q_id,data in teacher_answer_key_3marks.items()}
    df['teacher_answer']=df['question_no'].map(teacher_answer)
    df['score']=df.apply(
        lambda row:evaluation_short_answer(row['student_answer'],row['teacher_answer'],3),
        axis=1
    )
    logging.info("main dataframe is created with all data with each question")
    total_mark=len(teacher_answer_key_3marks)*3
except Exception as e:
    raise CustomException(e,sys)


try:
    
    logging.info("taking final detail of adding score of each student is initialized")
    df.to_csv('data/each_score_detail.csv')
    final_score_df=df.groupby('student_id')['score'].sum().reset_index()
    student_name={student['student_id']:student['student_name'] for student in student_answers_for_testing}

    final_score_df['name']=final_score_df['student_id'].map(student_name)

    final_score_df=final_score_df[['student_id','name','score']]

    final_score_df['result']=final_score_df['score'].apply(lambda x :'pass'if x>(total_mark*0.40) else 'fail')


    final_score_df.to_csv('data/final_score_detail.csv')
    logging.info("final dataframe is created with student result pass or fail")
except Exception as e:
    raise CustomException(e,sys)
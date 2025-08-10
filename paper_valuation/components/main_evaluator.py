from paper_valuation.exception.custom_exception import CustomException
from paper_valuation.logging.logger import logging
from paper_valuation.components.constant.valuation_data import student_answers_for_testing,teacher_answer_key_3marks
from paper_valuation.components.constant.valuation_data import teacher_long_answer_key,student_long_answer
from paper_valuation.components.valuation import evaluation_short_answer,evaluation_long_answer
import pandas as pd
import sys
from paper_valuation.components.util.main_utils import save_csv_file

def short_answer_data_preparation()->pd.DataFrame:
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
        return df
    except Exception as e:
        raise CustomException(e,sys)
    
    
def assign_score_short_answer(df:pd.DataFrame)->dict:
    
    df['score']=df.apply(
            lambda row:evaluation_short_answer(row['student_answer'],row['teacher_answer'],3),
            axis=1)
    logging.info("main dataframe is created with all data with each question")
    total_mark=len(teacher_answer_key_3marks)*3
    return {
        "df":df,
        "total_mark":total_mark
    }


def final_short_answer_valuation(df:pd.DataFrame,total_mark:int):
    try:
        
        logging.info("taking final detail of adding score of each student is initialized")
        save_csv_file(df,'short_answer_score.csv')
        
        final_score_df=df.groupby('student_id')['score'].sum().reset_index()
        student_name={student['student_id']:student['student_name'] for student in student_answers_for_testing}
        final_score_df['name']=final_score_df['student_id'].map(student_name)

        final_score_df=final_score_df[['student_id','name','score']]
        final_score_df['result']=final_score_df['score'].apply(lambda x :'pass'if x>(total_mark*0.40) else 'fail')

        save_csv_file(final_score_df,'short_answer_final.csv')
        logging.info("final dataframe is created with student result pass or fail")
    except Exception as e:
        raise CustomException(e,sys)
    
    
def long_answer_preparation()->dict:
    try:
        data=[]
        logging.info("long answer valuation is started")
        for student in student_long_answer:
            for q_id,answer in student['answers'].items():
                data.append({
                    "student_id":student['student_id'],
                    "student_name":student['student_name'],
                    "q_id":q_id,
                    "student_answer":answer
                })
        df=pd.DataFrame(data)
        keypoint_map={q_id:data['keypoints'] for q_id,data in teacher_long_answer_key.items()}
        total_mark_map={q_id:data['total_marks'] for q_id,data in teacher_long_answer_key.items()}
        total_mark=sum(int(data) for data in total_mark_map.values())
        df['keypoint']=df['q_id'].map(keypoint_map)
        df['total_mark']=df['q_id'].map(total_mark_map)
        logging.info("long answer valuation is done")
        
        return {
            "df":df,
            "total_mark":total_mark
        }
    except Exception as e:
        raise CustomException(e,sys)
    
def assign_score_long_answer(df:pd.DataFrame,total_mark:int)->dict:
    try:
        df['score']=df.apply(
        lambda row:evaluation_long_answer(row['student_answer'],row['keypoint'],row['total_mark']),
        axis=1
        )
        logging.info("long answer score is assigned")
        return {
            "df":df,
            "total_mark":total_mark
        }
    except Exception as e:
        raise CustomException(e,sys)
    
def final_long_answer_valuation(df:pd.DataFrame,total_mark:int):
    try:
        save_csv_file(df,'long_answer_score.csv')
        final_score_df=df.groupby('student_id')['score'].sum().reset_index()
        final_score_df['student_name']=df['student_id'].map(df['student_name'])
        final_score_df=final_score_df[['student_id','student_name','score']]
        final_score_df['result']=final_score_df['score'].apply(
            lambda x:'pass' if x>=(total_mark*0.5) else 'fail'
        )
        save_csv_file(final_score_df,'long_answer_final.csv')
        logging.info("long answer valuation is completely done")
    except Exception as e:
        raise CustomException(e,sys)
    
    
    
def start_main_evaluator():
    df=short_answer_data_preparation()
    report=assign_score_short_answer(df)
    final_short_answer_valuation(report['df'],report['total_mark'])
    
    
    report=long_answer_preparation()
    report=assign_score_long_answer(report['df'],report['total_mark'])
    final_long_answer_valuation(report['df'],report['total_mark'])
    
    
    
    
if __name__=="__main__":
    start_main_evaluator()
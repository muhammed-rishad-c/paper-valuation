from paper_valuation.exception.custom_exception import CustomException
from paper_valuation.logging.logger import logging
from paper_valuation.components.constant.valuation_data import student_answers_for_testing, teacher_answer_key_3marks
from paper_valuation.components.constant.valuation_data import teacher_long_answer_key, student_long_answer
from paper_valuation.components.valuation import evaluation_short_answer, evaluation_long_answer
import pandas as pd
import sys
from paper_valuation.components.util.main_utils import save_csv_file


def short_answer_data_preparation() -> pd.DataFrame:
    """
    Prepare short answer data for evaluation
    """
    flat_data = []
    try:
        logging.info("📊 Preparing short answer data...")
        
        for student in student_answers_for_testing:
            for q_id, answer in student['answers'].items():
                flat_data.append({
                    "student_id": student['student_id'],
                    "question_no": q_id,
                    "student_answer": answer
                })
        
        df = pd.DataFrame(flat_data)
        
        # Map teacher answers
        teacher_answer = {q_id: data['answer'] for q_id, data in teacher_answer_key_3marks.items()}
        df['teacher_answer'] = df['question_no'].map(teacher_answer)
        
        logging.info(f"✅ Prepared {len(df)} short answer rows for {len(student_answers_for_testing)} students")
        
        return df
        
    except Exception as e:
        raise CustomException(e, sys)


def assign_score_short_answer(df: pd.DataFrame) -> dict:
    """
    Evaluate all short answers and assign scores
    
    IMPROVED: Now uses threshold=0.45 (more lenient)
    """
    try:
        logging.info("🎯 Starting short answer evaluation...")
        
        # Evaluate each answer with improved threshold
        df['score'] = df.apply(
            lambda row: evaluation_short_answer(
                student_answer=row['student_answer'],
                teacher_answer=row['teacher_answer'],
                max_mark=3,
                threshold=0.45  # ← IMPROVED: More lenient threshold
            ),
            axis=1
        )
        
        logging.info("✅ Short answer evaluation completed")
        
        # Calculate total marks possible
        total_mark = len(teacher_answer_key_3marks) * 3
        
        return {
            "df": df,
            "total_mark": total_mark
        }
        
    except Exception as e:
        raise CustomException(e, sys)


def final_short_answer_valuation(df: pd.DataFrame, total_mark: int):
    """
    Generate final results and save to CSV
    """
    try:
        logging.info("📋 Generating final short answer results...")
        
        # Save detailed scores
        save_csv_file(df, 'short_answer_score.csv')
        
        # Calculate final scores per student
        final_score_df = df.groupby('student_id')['score'].sum().reset_index()
        
        # Add student names
        student_name = {student['student_id']: student['student_name'] for student in student_answers_for_testing}
        final_score_df['name'] = final_score_df['student_id'].map(student_name)
        
        # Reorder columns
        final_score_df = final_score_df[['student_id', 'name', 'score']]
        
        # Calculate percentage
        final_score_df['percentage'] = (final_score_df['score'] / total_mark * 100).round(2)
        
        # Determine pass/fail (40% passing threshold)
        final_score_df['result'] = final_score_df['score'].apply(
            lambda x: 'Pass' if x >= (total_mark * 0.40) else 'Fail'
        )
        
        # Save final results
        save_csv_file(final_score_df, 'short_answer_final.csv')
        
        logging.info("✅ Final short answer results saved")
        
        # Print summary
        print("\n" + "=" * 60)
        print("SHORT ANSWER RESULTS SUMMARY")
        print("=" * 60)
        print(final_score_df.to_string(index=False))
        print("=" * 60 + "\n")
        
    except Exception as e:
        raise CustomException(e, sys)


def long_answer_preparation() -> dict:
    """
    Prepare long answer data for evaluation
    """
    try:
        data = []
        
        logging.info("📊 Preparing long answer data...")
        
        for student in student_long_answer:
            for q_id, answer in student['answers'].items():
                data.append({
                    "student_id": student['student_id'],
                    "student_name": student['student_name'],
                    "q_id": q_id,
                    "student_answer": answer
                })
        
        df = pd.DataFrame(data)
        
        # Map key points and marks
        keypoint_map = {q_id: data['keypoints'] for q_id, data in teacher_long_answer_key.items()}
        total_mark_map = {q_id: data['total_marks'] for q_id, data in teacher_long_answer_key.items()}
        
        df['keypoint'] = df['q_id'].map(keypoint_map)
        df['total_mark'] = df['q_id'].map(total_mark_map)
        
        # Calculate total marks
        total_mark = sum(int(data) for data in total_mark_map.values())
        
        logging.info(f"✅ Prepared {len(df)} long answer rows")
        
        return {
            "df": df,
            "total_mark": total_mark
        }
        
    except Exception as e:
        raise CustomException(e, sys)


def assign_score_long_answer(df: pd.DataFrame, total_mark: int) -> dict:
    """
    Evaluate all long answers and assign scores
    
    IMPROVED: Now uses holistic_threshold=0.30, point_threshold=0.35 (more lenient)
    """
    try:
        logging.info("🎯 Starting long answer evaluation...")
        
        # Evaluate each answer with improved thresholds
        df['score'] = df.apply(
            lambda row: evaluation_long_answer(
                student_answer=row['student_answer'],
                teacher_answer=row['keypoint'],
                max_mark=row['total_mark'],
                holistic_threshold=0.30,  # ← IMPROVED: More lenient
                point_threshold=0.35       # ← IMPROVED: More lenient
            ),
            axis=1
        )
        
        logging.info("✅ Long answer evaluation completed")
        
        return {
            "df": df,
            "total_mark": total_mark
        }
        
    except Exception as e:
        raise CustomException(e, sys)


def final_long_answer_valuation(df: pd.DataFrame, total_mark: int):
    """
    Generate final results and save to CSV
    """
    try:
        logging.info("📋 Generating final long answer results...")
        
        # Save detailed scores
        save_csv_file(df, 'long_answer_score.csv')
        
        # Calculate final scores per student
        final_score_df = df.groupby('student_id')['score'].sum().reset_index()
        
        # Add student names (fix: use first() to avoid series mapping issues)
        student_names = df.groupby('student_id')['student_name'].first()
        final_score_df['student_name'] = final_score_df['student_id'].map(student_names)
        
        # Reorder columns
        final_score_df = final_score_df[['student_id', 'student_name', 'score']]
        
        # Calculate percentage
        final_score_df['percentage'] = (final_score_df['score'] / total_mark * 100).round(2)
        
        # Determine pass/fail (40% passing threshold for consistency)
        final_score_df['result'] = final_score_df['score'].apply(
            lambda x: 'Pass' if x >= (total_mark * 0.40) else 'Fail'
        )
        
        # Save final results
        save_csv_file(final_score_df, 'long_answer_final.csv')
        
        logging.info("✅ Final long answer results saved")
        
        # Print summary
        print("\n" + "=" * 60)
        print("LONG ANSWER RESULTS SUMMARY")
        print("=" * 60)
        print(final_score_df.to_string(index=False))
        print("=" * 60 + "\n")
        
    except Exception as e:
        raise CustomException(e, sys)


def start_main_evaluator():
    """
    Main evaluation pipeline
    """
    try:
        print("\n" + "=" * 60)
        print("🎓 PAPER VALUATION SYSTEM - MAIN EVALUATOR")
        print("=" * 60)
        
        # ==========================================
        # SHORT ANSWER EVALUATION
        # ==========================================
        print("\n📝 PHASE 1: SHORT ANSWER EVALUATION")
        print("-" * 60)
        
        df = short_answer_data_preparation()
        report = assign_score_short_answer(df)
        final_short_answer_valuation(report['df'], report['total_mark'])
        
        # ==========================================
        # LONG ANSWER EVALUATION
        # ==========================================
        print("\n📚 PHASE 2: LONG ANSWER EVALUATION")
        print("-" * 60)
        
        report = long_answer_preparation()
        report = assign_score_long_answer(report['df'], report['total_mark'])
        final_long_answer_valuation(report['df'], report['total_mark'])
        
        print("\n" + "=" * 60)
        print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n📁 Output files generated:")
        print("   - short_answer_score.csv (detailed scores)")
        print("   - short_answer_final.csv (final results)")
        print("   - long_answer_score.csv (detailed scores)")
        print("   - long_answer_final.csv (final results)")
        print("=" * 60 + "\n")
        
    except Exception as e:
        logging.error(f"❌ Evaluation failed: {str(e)}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    start_main_evaluator()
import os,sys
import pandas as pd
from paper_valuation.exception.custom_exception import CustomException
from paper_valuation.logging.logger import logging


def save_csv_file(file:pd.DataFrame,filename:str)->bool:
    try:
        dir_name='data'
        os.makedirs(dir_name,exist_ok=True)
        file.to_csv(f'{dir_name}/{filename}')
    except Exception as e:
        raise CustomException(e,sys)
    
from setuptools import find_packages,setup
from typing import List
from paper_valuation.exception.custom_exception import CustomException
import sys



def get_requirement()->List[str]:
    try:
        requirement_list:List[str]=[]
        with open('requirement.txt','r') as file:
            lines=file.readlines()
            for line in lines:
                requirement=line.strip()
            
                if requirement and requirement!='-e .':
                    requirement_list.append(requirement)
    except Exception as e:
        raise CustomException(e,sys)
    
    return requirement_list

setup(
    name="paper-valuation",
    author="muhammed-rishad-c",
    author_email="muhammed.risshad@gmail.com",
    version="0.0.0.0",
    packages=find_packages(),
    install_requires=get_requirement()
)
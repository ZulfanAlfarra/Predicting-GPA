from src.logger import logging
from src.exception import CustomException
from src.components.get_data import GetData
import sys

from zenml import step

@step
def pull_data(data_path: str):

    try:
        get_data = GetData()
        get_data.initiate_data(data_path)
    except Exception as e:
        raise CustomException(e, sys) 


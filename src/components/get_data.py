import os 
import sys
import pandas as pd
from typing import Tuple
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

@dataclass
class PullData:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


class GetData:
    def __init__(self):
        self.pull_data = PullData()
    
    def initiate_data(self, path: str)-> Tuple[str, str]:
        """
        Get data from the source and divide the data into train and test file
        """
        logging.info("Pull data from the source")
        try:
            data = pd.read_csv(path)
            logging.info("Read data as DataFrame")
            os.makedirs(os.path.dirname(self.pull_data.raw_data_path), exist_ok=True)
            data.to_csv(self.pull_data.raw_data_path, index=False, header=True)

            logging.info("Split data into train and test then save it")
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
            train_data.to_csv(self.pull_data.train_data_path, index=False, header=True)
            test_data.to_csv(self.pull_data.test_data_path, index=False, header=True)
            logging.info("Pulling data is completed")

            return self.pull_data.train_data_path, self.pull_data.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


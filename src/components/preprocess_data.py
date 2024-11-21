import os
import sys
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class DataStrategy(ABC):
    """
    Abstract classs for data strategy
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame):
        pass

class DataTransformation(DataStrategy):
    def handle_data(self, train_data_path:str, test_data_path:str) ->  Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
        try:
            logging.info("Making transformation pipeline")
            num_features = ['Study_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Social_Hours_Per_Day','Physical_Activity_Hours_Per_Day']
            cat_feature = ['Stress_Level']
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info("Transforming train and test data")
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_features),
                    ("cat_pipeline", cat_pipeline, cat_feature)
                ]
            )

            # Read train and test data from file paths
            logging.info(f"Reading train data from {train_data_path}")
            train_df = pd.read_csv(train_data_path)
            logging.info(f"Reading test data from {test_data_path}")
            test_df = pd.read_csv(test_data_path)

            # Check if the data is loaded properly
            if train_df.empty or test_df.empty:
                raise CustomException(f"Error: Train or test data is empty from the provided path.", sys)

            logging.info("Transforming train and test data")
            train_data_processed = preprocessor.fit_transform(train_df)
            test_data_processed = preprocessor.transform(test_df)

            logging.info("Saving preprocessor to artifacts")
            save_object(os.path.join('artifacts', 'preprocessor.pkl'), preprocessor)

            return train_data_processed, test_data_processed, preprocessor

        except Exception as e:
            CustomException(e, sys)
        
    
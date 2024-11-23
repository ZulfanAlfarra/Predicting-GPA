from src.components.data_processor import CustomData
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

import pandas as pd
import os
import sys

def predict(features: pd.DataFrame):
    try:
        logging.info("Load model and preprocessor")
        model = load_object(os.path.join("artifacts", "model.pkl"))
        preprocessor = load_object(os.path.join("artifacts", "preprocessor.pkl"))

        data_processed = preprocessor.transform(features)
        pred = model.predict(data_processed)

        return pred
    except Exception as e:
        raise CustomException(e, sys)
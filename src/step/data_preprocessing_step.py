from zenml import step
from src.components.data_processor import DataTransformation
from src.exception import CustomException
import sys
from typing import Tuple
from src.logger import logging
import numpy as np
import pandas as pd
from typing_extensions import Annotated
from zenml.materializers.numpy_materializer import NumpyMaterializer

@step(output_materializers={"train_arr": NumpyMaterializer, "test_arr": NumpyMaterializer})
def process_data(data_train_path:str, data_test_path: str) -> Tuple[
    Annotated[np.ndarray, "train_arr"], 
    Annotated[np.ndarray, "test_arr"]]:
    try:
        logging.info("Starting data preprocessing step")
        data_transformation= DataTransformation()
        train_arr, test_arr, _ = data_transformation.handle_data(data_train_path, data_test_path)
        
        return train_arr, test_arr
    except Exception as e:
        raise CustomException(e, sys)
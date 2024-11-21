from zenml import step
from src.components.preprocess_data import DataTransformation
from src.exception import CustomException
import sys
from typing import Tuple
from src.logger import logging
import numpy as np
import pandas as pd
from typing_extensions import Annotated
from zenml.materializers.numpy_materializer import NumpyMaterializer

@step(output_materializers={"train_data_preprocessed": NumpyMaterializer, "test_data_preprocessed": NumpyMaterializer})
def process_data(data_train_path:str, data_test_path: str) -> Tuple[
    Annotated[np.ndarray, "train_data_preprocessed"], 
    Annotated[np.ndarray, "test_data_preprocessed"]]:
    try:
        logging.info("Starting data preprocessing step")
        data_transformation= DataTransformation()
        train_data_processed, test_data_processed, _ = data_transformation.handle_data(data_train_path, data_test_path)


        # train_data_processed = pd.DataFrame(train_data_processed)
        # test_data_processed = pd.DataFrame(test_data_processed)
        
        return train_data_processed, test_data_processed
    except Exception as e:
        raise CustomException(e, sys)
import sys
import numpy as np

from zenml import step
from src.components.data_processor import DataSplitter
from src.logger import logging
from src.exception import CustomException
from typing import Tuple
from typing_extensions import Annotated
from zenml.materializers.numpy_materializer import NumpyMaterializer

@step(output_materializers={"X_train": NumpyMaterializer, "X_test": NumpyMaterializer, "y_train": NumpyMaterializer, "y_test": NumpyMaterializer})
def split_data(train_arr: np.ndarray, test_arr: np.ndarray) -> Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"],
]:
    try:
        logging.info("Starting split data")
        data_spliter = DataSplitter()
        X_train, X_test, y_train, y_test = data_spliter.handle_data(train_arr, test_arr)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise CustomException(e, sys)
    
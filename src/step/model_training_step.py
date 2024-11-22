import sys
import numpy as np
from src.exception import CustomException
from src.logger import logging
from typing import Tuple
from typing_extensions import Annotated
from src.components.model_trainer import ModelTrainer

from zenml import step

@step
def train_step(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Annotated[str, "model_path"]:
    try:
        logging.info("Start training model")
        model_trainer = ModelTrainer()
        model_path = model_trainer.engine(X_train, X_test, y_train, y_test)
        return model_path
    except Exception as e:
        raise CustomException(e, sys)
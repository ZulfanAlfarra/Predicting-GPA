from dataclasses import dataclass
import os
import sys

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import r2_score
from src.logger import logging
from src.utils import save_object
from src.exception import CustomException


@dataclass
class ModelTrainerConfig:
    model_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def engine(self, X_train, X_test, y_train, y_test) -> str:
        try:
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=0.1),
                'Lasso Regression': Lasso(alpha=0.1),
                'Random Forest': RandomForestRegressor()
                }
            
            result = {}
            logging.info("Training some models")
            for i in range(len(models)):
                model = list(models.values())[i]

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                test_score = r2_score(y_test, y_pred)

                result[list(models.keys())[i]] = test_score
            
            best_model_score= max(list(sorted(result.values())))
            logging.info(f"Best model score: {best_model_score}")

            best_model_name = list(result.keys())[list(result.values()).index(best_model_score)]
            logging.info(f"Best model name: {best_model_name}")

            best_model = models[best_model_name]

            logging.info("Save best model")
            save_object(self.model_trainer_config.model_path, best_model)

            return self.model_trainer_config.model_path
        except Exception as e:
            raise CustomException(e, sys)
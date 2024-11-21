from zenml import pipeline
from src.step.getting_data import pull_data
from src.step.preprocessing_data import process_data

@pipeline
def train_pipeline(data_path:str):
    data_train_path, data_test_path = pull_data(data_path)
    train_data_processed, test_data_processed = process_data(data_train_path, data_test_path)

from zenml import pipeline
from src.step.data_loading_step import pull_data
from src.step.data_preprocessing_step import process_data
from src.step.data_split_step import split_data

@pipeline
def train_pipeline(data_path:str):
    data_train_path, data_test_path = pull_data(data_path)
    train_arr, test_arr = process_data(data_train_path, data_test_path)
    X_train, X_test, y_train, y_test = split_data(train_arr, test_arr)

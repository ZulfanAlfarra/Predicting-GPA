from zenml import pipeline
from src.step.getting_data import pull_data

@pipeline
def train_pipeline(data_path:str):
    pull_data(data_path)
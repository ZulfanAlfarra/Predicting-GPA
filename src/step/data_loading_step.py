from src.exception import CustomException
from src.components.data_loader import GetData
import pandas as pd
import sys

from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from src.components.data_loader import GetData


@step
def pull_data(data_path: str) -> Tuple[
    Annotated[str, "data_train_path"], 
    Annotated[str, "data_test_path"]
    ]:

    try:
        get_data = GetData()
        data_train_path, data_test_path = get_data.initiate_data(data_path)
        return data_train_path, data_test_path
    except Exception as e:
        raise CustomException(e, sys) 


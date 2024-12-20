import pickle
import os
import sys

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        CustomException(e, sys)


def load_object(file_path):
        try:
            with open(file_path, 'rb') as file_obj:
                return pickle.load(file_obj)
        except Exception as e:
            raise CustomException(e, sys)
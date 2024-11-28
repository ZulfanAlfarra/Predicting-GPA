import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

list_of_files = [
    f"src/__init__.py",
    f"src/components/__init__py",
    f"src/pipelines/__init__py",
    f"src/step/__init__py",
    f"src/exception.py",
    f"src/logger.py",
    f"src/utils.py",
    f"notebook/trial.ipynb",
    f"templates/index.html",
    "app.py",
    "run_pipeline.py",
    "setup.py",
    "requirements.txt"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,'w') as f:
            logging.info(f"Creating empty file: {filepath}")
            pass
    
    else:
        logging.info(f"{filepath} is already exist")
            
    

import os
from pathlib import Path

project_name="src"

list_of_files=[

# core
f"{project_name}/__init__.py",

# components (aligned with your pipeline)
f"{project_name}/components/__init__.py",
f"{project_name}/components/data_ingestion.py",
f"{project_name}/components/data_transformation.py",
f"{project_name}/components/model_trainer.py",
f"{project_name}/components/model_evaluation.py",

# config
f"{project_name}/configuration/__init__.py",
f"{project_name}/configuration/config.py",

# entity (for configs + artifacts)
f"{project_name}/entity/__init__.py",
f"{project_name}/entity/config_entity.py",
f"{project_name}/entity/artifact_entity.py",

# pipeline
f"{project_name}/pipeline/__init__.py",
f"{project_name}/pipeline/training_pipeline.py",
f"{project_name}/pipeline/prediction_pipeline.py",

# utils
f"{project_name}/utils/__init__.py",
f"{project_name}/utils/logger.py",
f"{project_name}/utils/main_utils.py",

# exception
f"{project_name}/exception/__init__.py",
f"{project_name}/exception/exception.py",

# api
"app.py",

# infra
"requirements.txt",
"Dockerfile",
".dockerignore",

# configs
"config/model.yaml",
"config/schema.yaml",

# misc
"main.py",
"setup.py",
"pyproject.toml",
"README.md",
]

for filepath in list_of_files:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)

    if filedir!="":
        os.makedirs(filedir,exist_ok=True)

    if(not os.path.exists(filepath)) or(os.path.getsize(filepath)==0):
        with open(filepath,"w") as f:
            pass
    else:
        print(f"file already exists: {filepath}")
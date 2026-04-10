from dataclasses import dataclass
from typing import Any

@dataclass
class DataIngestionArtifact:
    X:Any
    y:Any

@dataclass
class DataTransformationArtifact:
    X_train:Any
    X_test:Any
    y_train:Any
    y_test:Any
    scaler:Any

@dataclass
class ModelTrainerArtifact:
    models:dict
    best_model_name:str
    best_model:Any

@dataclass
class ModelEvaluationArtifact:
    metrics:dict
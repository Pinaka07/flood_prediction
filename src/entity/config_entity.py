from dataclasses import dataclass

@dataclass
class TrainingConfig:
    data_path:str
    model_dir:str
    test_size:float
    random_state:int
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml
import os

@dataclass
class ModelConfig:
    lookahead_values: List[int]
    features: List[str]
    threshold_bounds: tuple
    trading_params: Dict
    
    @classmethod
    def from_yaml(cls, env: str = "development") -> "ModelConfig":
        config_path = os.path.join(
            os.path.dirname(__file__), 
            f"{env}.yaml"
        )
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)

@dataclass
class AppConfig:
    model: ModelConfig
    database: Dict
    api: Dict
    
    @classmethod
    def load(cls, env: str = "development") -> "AppConfig":
        return cls(
            model=ModelConfig.from_yaml(env),
            database=load_db_config(env),
            api=load_api_config(env)
        )
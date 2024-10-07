from dataclasses import dataclass
from omegaconf import MISSING
import torch.nn as nn

@dataclass
class OptimConfig:
    batch_size: int = MISSING
    learning_rate: float = MISSING
    weight_decay: float = MISSING

@dataclass
class SampleConfig:
    top_k: int = MISSING

@dataclass
class SystemConfig:
    input_file: str = MISSING
    work_dir: str = MISSING
    resume: bool = MISSING
    sample_only: bool = MISSING
    num_workers: int = MISSING
    max_steps: int = MISSING
    device: str = MISSING
    seed: int = MISSING
    
@dataclass
class Config:
    model: nn.Module = MISSING
    optimization: OptimConfig = OptimConfig()
    system: SystemConfig = SystemConfig()
    sampling: SampleConfig = SampleConfig()
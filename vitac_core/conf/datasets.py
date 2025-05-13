"""
datasets.py

Base Hydra Structured Config for defining various pretraining datasets and appropriate configurations. Uses a simple,
single inheritance structure.
"""
from dataclasses import dataclass
from typing import Any, Tuple, List
from dataclasses import field

from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import MISSING
import os
rootPath = os.getcwd()



@dataclass
class DatasetConfig:
    name: str = MISSING
    # remarks: str = MISSING
    path: str = MISSING
    artifact_path: str = MISSING

    # Streaming Parameters (assumes fully preprocessed dataset lives at `stream_prefix/...`)
    #   =>> Deprecated as of `v2`
    stream: bool = True
    stream_prefix: str = "data/processed"

    # Dataset-Specific Parameters
    resolution: int = 224
    normalization: Tuple[Any, Any] = MISSING

    # data_formats: List[Any] = MISSING
    # For preprocessing --> maximum size of saved frames (assumed square)
    preprocess_resolution: int = MISSING
    preprocess_resolution_width: int = MISSING

    # Validation Parameters
    train_val_ratio: float = MISSING

    # Language Modeling Parameters
    language_model: str = "distilbert-base-uncased"
    hf_cache: str = "data/hf-cache"

    # Language Modeling Parameters
    max_lang_len: int = MISSING

    # Dataset sets the number of pretraining epochs (general rule :: warmup should be ~5% of full)
    warmup_epochs: int = MISSING
    max_epochs: int = MISSING

    # Plausible Formats --> These are instantiations each "batch" could take, with a small DSL
    #   > Note: Assumes final element of the list is the "most expressive" --> used to back-off
    batch_formats: Any = (
        ("only_in", ("state_i",)),
        # ("before+in", ("state_initial", "state_i")),
        #("before+3in+final", ("state_initial", "state_i", "state_j", "state_k", "state_final")),
        # ("fk4_2f_v-sim", ("state_i", "state_j")),
        # ("fk4_2f_vt-sim", ("state_i", "state_j")),

    )

    # Preprocessing :: Frame-Sampling Parameters
    initial_final_alpha: float = 0.2
@dataclass
class VTDexManipConfig(DatasetConfig):
    # fmt: off
    name: str = "VTDexManip"
    remarks: str = ""
    path: str = rootPath+"/data/raw/vt-dex-manip"
    artifact_path: str = rootPath+"/data/processed/vt-dex-manip/"

    # Dataset Specific arguments
    normalization: Tuple[Any, Any] = (                              # Mean & Standard Deviation (default :: ImageNet)
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )

    # data formats
    data_formats: List[Any] = field(default_factory=lambda: [
        "vision",
        "tactile"
    ])

    # Sth-Sth-v2 Videos have a fixed height of 240; we'll crop to square at this resolution!
    preprocess_resolution: int = 240
    preprocess_resolution_width: int = 424

    # Validation Parameters
    train_val_ratio: float = 5                                        # Number of Validation Clips (fast evaluation!)

    # Epochs for Dataset
    warmup_epochs: int = 20
    max_epochs: int = 400

    # Language Modeling Parameters
    max_lang_len: int = 20

    # filter
    # fmt: on


# Create a configuration group `dataset` and populate with the above...
#   =>> Note :: this is meant to be extendable --> add arbitrary datasets & mixtures!
cs = ConfigStore.instance()
# cs.store(group="dataset", name="ViTacReal", node=ViTacRealConfig)
cs.store(group="dataset", name="VTDexManip", node=VTDexManipConfig)


DatasetList: List[str] = ["VTDexManip"]
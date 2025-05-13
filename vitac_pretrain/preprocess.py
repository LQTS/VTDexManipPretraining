"""
preprocess.py

Centralized script for preprocessing various video/vision-language datasets for GPU pretraining, using a multi-stage,
multiprocessing approach.

Run as a standalone script, *prior* to calling `pretrain.py` =>> mostly because we want to preprocess the data once, as
a fixed cost.
"""
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from vitac_core.conf import DatasetConfig
from vitac_core.overwatch import OverwatchRich
from vitac_core.preprocessing import extract_frames, preprocess_language, extract_tactile_data, unify_batches
from vitac_core.util import set_global_seed



@dataclass
class PreprocessingConfig:
    # fmt: off
    defaults: List[Any] = field(default_factory=lambda: ["_self_",
                                                         {"dataset": "VTDexManip"},
                                                         {"override hydra/job_logging": "overwatch_rich"}
                                                         ])
    hydra: Dict[str, Any] = field(
        default_factory=lambda: {"run": {"dir": "./vitac_pretrain/runs/preprocessing/${now:%m-%d}/dataset-${dataset.name}"}}
    )

    data_modality: List[Any] = field(default_factory=lambda:["vision", "tactile"])


    if_rewrite: bool = False

    tac_trigger_threshold: float = 0.2

    if_debug: bool = False

    # Command Line Arguments
    seed: int = 42                                  # Random Seed (for reproducibility)

    # Composable / Structured Arguments
    dataset: DatasetConfig = MISSING                # Dataset(s) for pretraining/preprocessing
    # fmt: on


# Hydra Setup :: Retrieve ConfigStore (Singleton) & Register Components
cs = ConfigStore.instance()
cs.store(group="hydra/job_logging", name="overwatch_rich", node=OverwatchRich)
cs.store(name="config", node=PreprocessingConfig)

@hydra.main(config_path=None, config_name="config")
def preprocess(cfg: PreprocessingConfig) -> None:
    # Grab Logger
    overwatch = logging.getLogger(__file__)
    overwatch.info("Preprocessing :: Running Phases for Frame Extraction, Language Compilation, and Batching...")

    # Set Randomness
    set_global_seed(cfg.seed)

    assert "vision" in cfg.data_modality, "Vision modality is necessary in this Version"

    unify_t_dir, unify_v_dir = dict(), dict()
    multimodel_input = dict()
    # Phase 1 :: Serialize Frames from Video Clips --> get `registry` (index files) for train and validation
    train_registry, val_registry, train_dir, val_dir = extract_frames(
        cfg.dataset.name,
        path=cfg.dataset.path,
        artifact_path=cfg.dataset.artifact_path,
        preprocess_resolution=cfg.dataset.preprocess_resolution,
        train_val_ratio=cfg.dataset.train_val_ratio,
        rewrite=cfg.if_rewrite,
    )
    unify_t_dir["vision"] = train_dir
    unify_v_dir["vision"] = val_dir


    index_dir = Path(cfg.dataset.artifact_path) / cfg.dataset.name / "index"

    # Phase 2 :: Extract & Normalize & Binary Tactile Data, serialize Frames align to Video Clips
    #            --> get `registry` (index files) for train and validation
    if "tactile" in cfg.data_modality:
        (
            unify_t_dir["tactile"],
            unify_v_dir["tactile"],
            multimodel_input["tac_triggered"]
        ) = extract_tactile_data(
            cfg.dataset.name,
            train_registry,
            val_registry,
            tac_trig_thre=cfg.tac_trigger_threshold,
            path=cfg.dataset.path,
            artifact_path=cfg.dataset.artifact_path,
            rewrite=cfg.if_rewrite,
            debug_mode=cfg.if_debug
        )

    # Phase 3 :: Assemble "Data-Locked" Batch Sets for Various Models (e.g., for single-frame/dual-frame/quintet)

    unify_batches(
        cfg.dataset.name,
        train_registry,
        val_registry,
        unify_t_dir,
        unify_v_dir,
        index_dir,
        multimodel_input,
        data_modality=cfg.data_modality,
        batch_formats=cfg.dataset.batch_formats,
        max_epochs=cfg.dataset.max_epochs,
        initial_final_alpha=cfg.dataset.initial_final_alpha,
        rewrite=cfg.if_rewrite
    )

    overwatch.info("Preprocessing Complete!")


if __name__ == "__main__":
    preprocess()

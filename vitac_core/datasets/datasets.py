"""
datasets.py

Core Pytorch Dataset implementations for the various "data flavors" used by the different representation learning models
(Voltron and data-locked reproductions). Crucially, ach dataset loads from the corresponding serialized batch files that
define the exact data (and order of iterating through the data) to see during each epoch.

Notably, these serialized files control exactly what data is seen by *all* methods *across epochs*; using these files is
critical to reproducibility & comparisons.

The file contains logic for a "standard" Dataset; all files (batch index files, image/video/language files) are stored
on local disk, assuming storage conducive to fast random reads. For a "streaming" dataset (loading data directly from
GCP Buckets/Amazon S3), see `v1/stream_datasets.py`.
"""
from pathlib import Path
from typing import Any, Optional, Tuple, Union, Callable, List

import h5py
import pickle
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose

from vitac_core.preprocessing.transforms import get_online_transform, get_online_tactile_transform
from vitac_core.conf.datasets import DatasetList


class PretrainDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.epoch, self.h5, self.vid, self.states = 0, None, None, None
        # self.index_path= None
        self.index_path, self.language_path, self.language = None, None, None

    def hydrate(self, path: Path) -> None:
        # Create Open HDF5 Handle
        self.h5 = h5py.File(path, "r")
        self.vid, self.states = self.h5["vid"].asstr(), self.h5["states"].asstr()
        # Create Tactile Handle from HDF5
        if "tactile" in list(self.h5.keys()):
            self.tactile = self.h5["tactile"].asstr()
        else:
            self.tactile = None

        # Load Language Index
        if self.language_path is not None:
            self.language = torch.load(self.index_path / self.language_path)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError("PretrainDataset is an abstract class; should never be initialized directly!")

    def __len__(self) -> int:
        raise NotImplementedError("PretrainDataset is an abstract class; should never be initialized directly!")


class SingleViTacDataset(PretrainDataset):
    def __init__(
            self,
            epoch: int,
            img_transform: Compose,
            tactile_preprocess_type: str,
            tactile_transform: Callable[[List[List[float]]], torch.Tensor],
            index_path: Path,
            data_modality: str,
            is_val: bool = False,
    ) -> None:
        super().__init__()
        self.index_path, self.is_val, self.val_loaded = index_path, is_val, False
        self.epoch, self.img_transform, self.hdf5_path, self.n_examples = epoch, img_transform, None, None
        self.data_modality = data_modality
        # for tactile methods
        self.tactile_transform = tactile_transform
        self.tactile_preprocess_type = tactile_preprocess_type

        # === Retrieve Epoch Batches --> only call before/between epochs (as we create new DataLoaders) ===
        self.set_epoch(epoch)

    def set_epoch(self, epoch: int) -> None:
        # Load Validation Batches
        if self.is_val and not self.val_loaded:
            self.hdf5_path = self.index_path / self.data_modality / "validation-batches.hdf5"
            with h5py.File(self.hdf5_path, "r") as h5:
                self.n_examples = len(h5["states"])

            # Set `val_loaded`
            self.val_loaded = True

        # Load Train Batches
        elif not self.is_val:
            self.hdf5_path = self.index_path / self.data_modality / f"train-epoch={epoch}-batches.hdf5"
            with h5py.File(self.hdf5_path, "r") as h5:
                self.n_examples = len(h5["states"])

        # # Assemble Dropout Indices
        # n_drop = int(self.lang_dropout * self.n_examples)
        # self.dropout_idxs = set(np.random.choice(self.n_examples, n_drop, replace=False))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return processed image frame and language, decomposed as the input_ids and attention_mask."""
        if self.h5 is None:
            self.hydrate(self.hdf5_path)

        # Get Vid ID --> parse out language, transform frame!
        vid = self.vid[idx]

        # Retrieve Frame & Return
        img = self.img_transform(read_image(str(self.index_path.parent / self.states[idx][0])))

        # Retrieve Tactile Clip & Return
        tac = self.tactile_transform(self.__Read_Tactile(idx))

        return img, tac

    def __Read_Tactile(self, idx: int) -> List[List[float]]:

        tactile_file_path = self.index_path.parent / self.tactile[idx][0]
        tactile_path = str(tactile_file_path).replace('_method_', self.tactile_preprocess_type)

        with open(tactile_path, "rb") as f:
            tactile_frame = pickle.load(f)
        return tactile_frame

    def __len__(self) -> int:
        return self.n_examples

        # cfg.model.data_modality,
        # datasets_input_config,


class SingleViTacLangDataset(PretrainDataset):
    def __init__(
            self,
            epoch: int,
            img_transform: Compose,
            tactile_preprocess_type: str,
            tactile_transform: Callable[[List[List[float]]], torch.Tensor],
            index_path: Path,
            data_modality: str,
            is_val: bool = False,
    ) -> None:
        super().__init__()
        self.index_path, self.is_val, self.val_loaded = index_path, is_val, False
        self.epoch, self.img_transform, self.hdf5_path, self.n_examples = epoch, img_transform, None, None

        self.data_modality = data_modality
        # for tactile methods
        self.tactile_transform = tactile_transform
        self.tactile_preprocess_type = tactile_preprocess_type

        self.lang_dropout, self.dropout_idxs = 0.0, set()
        # Set Language Path
        self.language_path = "val-language-index.pt" if self.is_val else "train-language-index.pt"
        # === Retrieve Epoch Batches --> only call before/between epochs (as we create new DataLoaders) ===
        self.set_epoch(epoch)

    def set_epoch(self, epoch: int) -> None:
        # Load Validation Batches
        if self.is_val and not self.val_loaded:
            self.hdf5_path = self.index_path / self.data_modality / "validation-batches.hdf5"
            with h5py.File(self.hdf5_path, "r") as h5:
                self.n_examples = len(h5["states"])

            # Set `val_loaded`
            self.val_loaded = True

        # Load Train Batches
        elif not self.is_val:
            self.hdf5_path = self.index_path / self.data_modality / f"train-epoch={epoch}-batches.hdf5"
            with h5py.File(self.hdf5_path, "r") as h5:
                self.n_examples = len(h5["states"])

        # # Assemble Dropout Indices
        n_drop = int(self.lang_dropout * self.n_examples)
        self.dropout_idxs = set(np.random.choice(self.n_examples, n_drop, replace=False))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return processed image frame and language, decomposed as the input_ids and attention_mask."""
        if self.h5 is None:
            self.hydrate(self.hdf5_path)

        # Get Vid ID --> parse out language, transform frame!
        vid = self.vid[idx]
        lang, lang_mask = self.language[vid]["input_ids"], self.language[vid]["attention_mask"]
        # Dropout Language (Naive Zeroing leads to NaN --> just want the "CLS" token)
        if idx in self.dropout_idxs:
            # Initial language token is *always* <CLS> = `101` --> last token always <SEP> = `102`
            lang[1:] *= 0
            lang_mask[1:] *= 0
        # Retrieve Frame & Return
        img = self.img_transform(read_image(str(self.index_path.parent / self.states[idx][0])))

        # Retrieve Tactile Clip & Return
        tac = self.tactile_transform(self.__Read_Tactile(idx))

        return img, tac, lang, lang_mask

    def __Read_Tactile(self, idx: int) -> List[List[float]]:

        tactile_file_path = self.index_path.parent / self.tactile[idx][0]
        tactile_path = str(tactile_file_path).replace('_method_', self.tactile_preprocess_type)

        with open(tactile_path, "rb") as f:
            tactile_frame = pickle.load(f)
        return tactile_frame

    def __len__(self) -> int:
        return self.n_examples


class DoubleViTacDataset(PretrainDataset):
    def __init__(
            self,
            epoch: int,
            img_transform: Compose,
            tactile_preprocess_type: str,
            tactile_transform: Callable[[List[List[float]]], torch.Tensor],
            index_path: Path,
            data_modality: str,
            is_val: bool = False,
    ) -> None:
        super().__init__()
        self.index_path, self.is_val, self.val_loaded = index_path, is_val, False
        self.epoch, self.img_transform, self.hdf5_path, self.n_examples = epoch, img_transform, None, None
        self.data_modality = data_modality
        # for tactile methods
        self.tactile_transform = tactile_transform
        self.tactile_preprocess_type = tactile_preprocess_type

        # === Retrieve Epoch Batches --> only call before/between epochs (as we create new DataLoaders) ===
        self.set_epoch(epoch)

    def set_epoch(self, epoch: int) -> None:
        # Load Validation Batches
        if self.is_val and not self.val_loaded:
            self.hdf5_path = self.index_path / self.data_modality / "validation-batches.hdf5"
            with h5py.File(self.hdf5_path, "r") as h5:
                self.n_examples = len(h5["states"])

            # Set `val_loaded`
            self.val_loaded = True

        # Load Train Batches
        elif not self.is_val:
            self.hdf5_path = self.index_path / self.data_modality / f"train-epoch={epoch}-batches.hdf5"
            with h5py.File(self.hdf5_path, "r") as h5:
                self.n_examples = len(h5["states"])

        # # Assemble Dropout Indices
        # n_drop = int(self.lang_dropout * self.n_examples)
        # self.dropout_idxs = set(np.random.choice(self.n_examples, n_drop, replace=False))

    def hydrate(self, path: Path) -> None:
        # Create Open HDF5 Handle
        self.h5 = h5py.File(path, "r")
        self.vid, self.states = self.h5["vid"].asstr(), self.h5["states"].asstr()
        # Create Tactile Handle from HDF5
        if "tactile" in list(self.h5.keys()):
            self.tactile = self.h5["tactile"].asstr()
        else:
            self.tactile = None

        if "sim_label" in list(self.h5.keys()):
            self.sim_label = self.h5["sim_label"]
        else:
            self.sim_label = None

        # Load Language Index
        if self.language_path is not None:
            self.language = torch.load(self.index_path / self.language_path)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Return processed image frame and language, decomposed as the input_ids and attention_mask."""
        if self.h5 is None:
            self.hydrate(self.hdf5_path)

        # Get Vid ID --> parse out language, transform frame!
        vid = self.vid[idx]

        # Retrieve Frame & Return
        imgs = self.states[idx]
        imgs = torch.stack([self.img_transform(read_image(str(self.index_path.parent / fn))) for fn in imgs])
        # img = self.img_transform(read_image(str(self.index_path.parent / self.states[idx][0])))

        # Retrieve Tactile Clip & Return
        tactiles = self.tactile[idx]
        tactiles = torch.Tensor([self.__Read_Tactile(
            str(self.index_path.parent / tac_path).replace('_method_', self.tactile_preprocess_type)) for tac_path in
                                 tactiles])

        sim_label = torch.Tensor([1])
        if self.sim_label is not None:
            sim_label = torch.Tensor(self.sim_label[idx])

        return imgs, tactiles, sim_label

    def __Read_Tactile(self, tactile_path: str) -> List[List[float]]:


        with open(tactile_path, "rb") as f:
            tactile_frame = pickle.load(f)
        return tactile_frame

    def __len__(self) -> int:
        return self.n_examples

        # cfg.model.data_modality,
        # datasets_input_config,
class Frame5Dataset(PretrainDataset):
    # the same as DoubleDataset
    def __init__(
            self,
            epoch: int,
            img_transform: Compose,
            tactile_preprocess_type: str,
            tactile_transform: Callable[[List[List[float]]], torch.Tensor],
            index_path: Path,
            data_modality: str,
            is_val: bool = False,
    ) -> None:
        super().__init__()
        self.index_path, self.is_val, self.val_loaded = index_path, is_val, False
        self.epoch, self.img_transform, self.hdf5_path, self.n_examples = epoch, img_transform, None, None
        self.data_modality = data_modality
        # for tactile methods
        self.tactile_transform = tactile_transform
        self.tactile_preprocess_type = tactile_preprocess_type

        # === Retrieve Epoch Batches --> only call before/between epochs (as we create new DataLoaders) ===
        self.set_epoch(epoch)

    def set_epoch(self, epoch: int) -> None:
        # Load Validation Batches
        if self.is_val and not self.val_loaded:
            self.hdf5_path = self.index_path / self.data_modality / "validation-batches.hdf5"
            with h5py.File(self.hdf5_path, "r") as h5:
                self.n_examples = len(h5["states"])

            # Set `val_loaded`
            self.val_loaded = True

        # Load Train Batches
        elif not self.is_val:
            self.hdf5_path = self.index_path / self.data_modality / f"train-epoch={epoch}-batches.hdf5"
            with h5py.File(self.hdf5_path, "r") as h5:
                self.n_examples = len(h5["states"])

        # # Assemble Dropout Indices
        # n_drop = int(self.lang_dropout * self.n_examples)
        # self.dropout_idxs = set(np.random.choice(self.n_examples, n_drop, replace=False))

    def hydrate(self, path: Path) -> None:
        # Create Open HDF5 Handle
        self.h5 = h5py.File(path, "r")
        self.vid, self.states = self.h5["vid"].asstr(), self.h5["states"].asstr()
        # Create Tactile Handle from HDF5
        if "tactile" in list(self.h5.keys()):
            self.tactile = self.h5["tactile"].asstr()
        else:
            self.tactile = None

        if "sim_label" in list(self.h5.keys()):
            self.sim_label = self.h5["sim_label"]
        else:
            self.sim_label = None

        # Load Language Index
        if self.language_path is not None:
            self.language = torch.load(self.index_path / self.language_path)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Return processed image frame and language, decomposed as the input_ids and attention_mask."""
        if self.h5 is None:
            self.hydrate(self.hdf5_path)

        # Get Vid ID --> parse out language, transform frame!
        vid = self.vid[idx]

        # Retrieve Frame & Return
        imgs = self.states[idx]
        imgs = torch.stack([self.img_transform(read_image(str(self.index_path.parent / fn))) for fn in imgs])
        # img = self.img_transform(read_image(str(self.index_path.parent / self.states[idx][0])))

        # Retrieve Tactile Clip & Return
        tactiles = self.tactile[idx]
        tactiles = torch.Tensor([self.__Read_Tactile(
            str(self.index_path.parent / tac_path).replace('_method_', self.tactile_preprocess_type)) for tac_path in
            tactiles])

        sim_label = torch.Tensor([1])
        if self.sim_label is not None:
            sim_label = torch.Tensor(self.sim_label[idx])

        return imgs, tactiles, sim_label

    def __Read_Tactile(self, tactile_path: str) -> List[List[float]]:

        # 读取文件
        with open(tactile_path, "rb") as f:
            tactile_frame = pickle.load(f)
        return tactile_frame

    def __len__(self) -> int:
        return self.n_examples

def get_datasets(
        epoch: int,
        dataset_name: str,
        model_arch: str,
        artifact_path: str,
        data_modality: str,
        data_formats: List[str],
        input_config: dict
) -> Tuple[PretrainDataset, PretrainDataset]:
    resolution = input_config["resolution"]
    normalization = input_config["normalization"]
    index = Path(artifact_path) / dataset_name / "index"
    img_transform = get_online_transform(dataset_name, model_arch, resolution, normalization)

    if "tactile" in data_formats:
        tactile_type = input_config["tactile_type"]
        tactile_transform = get_online_tactile_transform(dataset_name, model_arch)

    # Switch on `data_modality` --> differs based on `model_arch` (e.g., MVP --> img, V-Cond --> img, language)
    if dataset_name in DatasetList:
        if data_modality == "only_in":# and data_formats == ['vision', 'tactile']:
            train_ds = SingleViTacDataset(epoch, img_transform, tactile_type, tactile_transform, index, data_modality)
            val_ds = SingleViTacDataset(epoch, img_transform, tactile_type, tactile_transform, index, data_modality,
                                        is_val=True)
        elif data_modality in ['fk4_2f_v-sim', 'fk4_2f_vt-sim']:
            train_ds = DoubleViTacDataset(epoch, img_transform, tactile_type, tactile_transform, index, data_modality)
            val_ds = DoubleViTacDataset(epoch, img_transform, tactile_type, tactile_transform, index, data_modality,
                                        is_val=True)
        elif data_modality == "before+3in+final":
            train_ds = Frame5Dataset(epoch, img_transform, tactile_type, tactile_transform, index, data_modality)
            val_ds = Frame5Dataset(epoch, img_transform, tactile_type, tactile_transform, index, data_modality,
                                        is_val=True)
        else:
            raise ValueError(f"Data Modality `{data_modality}` for DataSet '{dataset_name}' is not supported!")
    else:
        raise ValueError(f"DataSet `{data_modality}` is not supported!")

    return train_ds, val_ds

import tarfile
from pathlib import Path

# checke the downloaded dataset
from tqdm import tqdm

dataset_tar_path = Path("data/VTDexManip_dataset")
tar_files = dataset_tar_path.glob("*.tar.gz")
assert len(list(tar_files))==7, f"num of tar.gz files is not 7, check whether the dataset files are all downloaded into the path {dataset_tar_path}"
img_tar_files = list(dataset_tar_path.glob("sub*"))
assert len(img_tar_files) == 5
tac_tar_file = dataset_tar_path / "tactile.tar.gz"
assert tac_tar_file.exists()
info_tar_file = dataset_tar_path / "info.tar.gz"
assert info_tar_file.exists()


# make raw data folder
raw_data_path = Path("data/raw")
raw_data_path.mkdir(exist_ok=True)
dataset_root_path = raw_data_path / "vt-dex-manip"
dataset_root_path.mkdir(exist_ok=True)
img_path = dataset_root_path / "videos"
# tac_path = dataset_root_path / "tactile"
# info_path = dataset_root_path / "info"
img_path.mkdir(exist_ok=True)
# tac_path.mkdir(exist_ok=True)
# info_path.mkdir(exist_ok=True)

# extract files
tac = tarfile.open(tac_tar_file)
tac.extractall(dataset_root_path)
print("extracting tactile data is done!")
info = tarfile.open(info_tar_file)
info.extractall(dataset_root_path)
print("extracting info data is done!")
for img_tar_file in tqdm(img_tar_files, desc="extract image data"):
    img = tarfile.open(img_tar_file)
    img.extractall(img_path)
print("extracting image data is done!")


# Dataset and Pretraining for VTDexManip

The code repository is for the ***VTDexManip*** dataset and pretraining, built upon [Voltron](https://github.com/siddk/voltron-robotics). These scripts can: 

- preprocess the dataset;
- pretrain the [benchmark](https://github.com/LQTS/VTDexManip) models (including **VT-JointPretrain**, **V-Pretrain** and **T-Pretrain**);
- visualize the dataset.

# **Dependencies**

( Follow the [instructions](https://github.com/LQTS/VTDexManip?tab=readme-ov-file#dependencies) )

# Dataset Preparation

We have released the dataset and anyone can access the dataset through the [link](https://1drv.ms/f/c/9054151f0ba654c9/EslUpgsfFVQggJB7AQAAAAABSp31p8Fft1mtHyOwJNqmoA) . The password is **vtdexmanip**

Put the downloaded dataset folder into "**data/**", and the file tree of "**data/**"will look like this:

```
data
├── tools
│   └── dataset_preparation.py
└── VTDexManip_dataset
    ├── info.tar.gz
    ├── License.txt
    ├── sub1.tar.gz
    ├── sub2.tar.gz
    ├── sub3.tar.gz
    ├── sub4.tar.gz
    ├── sub5.tar.gz
    └── tactile.tar.gz

```

To decompress the data, use

```bash
python data/tools/dataset_preparation.py
```

# Dataset Preprocess

The code [*vitac_pretrain/preprocess.py*](vitac_pretrain/preprocess.py)  is used to crop the images and binary the tactile signals.

```python
python vitac_pretrain/preprocess.py
```

# Model Pretraining

The config file is [*vitac_pretrain/ConfigBank.py*](vitac_pretrain/ConfigBank.py).

```
@dataclass
class Pretrain_Config:
    # load model training config
    model_dataset = VT20_PretrainConfig.VT20T_ReAll_TMR05_Bin_FT_CLS_ViTacReal
    # model_dataset = V_PretrainConfig.V_RePic_Bin_CLS_ViTacReal
    # model_dataset = T20_PretrainConfig.T20_ReTac_TMR05_Bin_FT_CLS_ViTacReal
    model_dataset[2]["dataset"] = "VTDexManip"
    model_dataset[3]["accelerator"] = "torchmulti" #torchone, torchmulti
```

Change ***model_dataset*** and run the command to train the models

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes 1 --nproc-per-node 2 vitac_pretrain/pretrain.py
```

If you want to run on **the single gpu**, change the config file:

```bash
model_dataset[3]["accelerator"] = "torchone"
```

and then run the command:

```bash
python vitac_pretrain/pretrain.py
```

# Visualize the dataset

To visualize the dataset trajectories, you can use

```python
python vitac_pretrain/visualize.py
```

You can change ***traj_id in the code***  to visualize the different manipulation trajectory.

# Contact

If you have any questions or need support, please contact <a href="mailto:l_qingtao@zju.edu.cn"> Qingtao Liu</a> or <a href="mailto:qi.ye@zju.edu.cn">Qi Ye</a>.
.

# BibTeX
```
@inproceedings{
liu2025vtdexmanip,
title={VTDexManip: A Dataset and Benchmark for Visual-tactile Pretraining and Dexterous Manipulation with Reinforcement Learning},
author={Qingtao Liu and Yu Cui and Zhengnan Sun and Gaofeng Li and Jiming Chen and Qi Ye},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=jf7C7EGw21}
}
```
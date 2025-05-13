import os.path
from dataclasses import dataclass


@dataclass
class VT20_PretrainConfig:
    shared_config = [
        {"dataset": "VTDexManip"},
        {"accelerator": "torchone"},
        {"tracking": "vitac-track"},
        {"override hydra/job_logging": "overwatch_rich"},
    ]
    VT20T_ReAll_TMR05_Bin_FT_ViTacReal = ["_self_", {"model": "vt20t-reall-tmr05-bin-ft"}] + shared_config
    VT20T_ReAll_TMR05_Bin_FT_CLS_ViTacReal = ["_self_", {"model": "vt20t-reall-tmr05-bin-ft-cls"}] + shared_config


@dataclass
class V_PretrainConfig:
    shared_config = [
        {"dataset": "VTDexManip"},
        {"accelerator": "torchone"},
        {"tracking": "vitac-track"},
        {"override hydra/job_logging": "overwatch_rich"},
    ]
    V_RePic_Bin_ViTacReal = ["_self_", {"model": "v-repic-bin-ft"}] + shared_config
    V_RePic_Bin_CLS_ViTacReal = ["_self_", {"model": "v-repic-bin-ft-cls"}] + shared_config


@dataclass
class T20_PretrainConfig:
    shared_config = [
        {"dataset": "VTDexManip"},
        {"accelerator": "torchone"},
        {"tracking": "vitac-track"},
        {"override hydra/job_logging": "overwatch_rich"},
    ]
    T20_ReTac_TMR05_Bin_FT_ViTacReal = ["_self_", {"model": "t20-retac-tmr05-bin-ft"}] + shared_config
    T20_ReTac_TMR05_Bin_FT_CLS_ViTacReal = ["_self_", {"model": "t20-retac-tmr05-bin-ft-cls"}] + shared_config

@dataclass
class Pretrain_Config:

    # load model training config
    model_dataset = VT20_PretrainConfig.VT20T_ReAll_TMR05_Bin_FT_CLS_ViTacReal
    # model_dataset = V_PretrainConfig.V_RePic_Bin_CLS_ViTacReal
    # model_dataset = T20_PretrainConfig.T20_ReTac_TMR05_Bin_FT_CLS_ViTacReal
    model_dataset[2]["dataset"] = "VTDexManip"
    model_dataset[3]["accelerator"] = "torchmulti" #torchone, torchmulti
    # configure work path
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #"/remote-home/share/lqt/touch_vision_manipulation/"
    hydra_ = {"run": {"dir": "vitac_pretrain/runs/train/binary/${model.identifier}+dataset-${dataset.name}"}}
    recon_pic_save_path = "reconstruction/"
    save_recons_freq = 20
    # whether to resume models
    resume = False

    # load model(checkpoint) to finetune
    if_finetone = False  ###################
    finetone_model_path = root_dir + "data/model_cache/v-cond+vit-small+sth-sth+epoch-400.pt"###################

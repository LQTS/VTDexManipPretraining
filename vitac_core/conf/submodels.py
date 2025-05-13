
from vitac_core.conf.models import (VTTRePicConfig, VRePicConfig,
                                       VTTReTacConfig, VTTReAllConfig,
                                       TReTacConfig, VReAllConfig, VReTacConfig, R3MSmallConfig)
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

@dataclass
class VT20TReAll_TMR05_Bin_FT_Config(VTTReAllConfig):
    identifier = "vt20t-reall-tmr05-bin-ft-w1-w10-bs64"
    remarks = "mr075-fr-bce-norelu"

    use_cls_token = False
    mask_ratio = 0.75
    tactile_mask_ratio = 0.5

    tactile_set_to_ZERO = False
    tactile_type = "binary_0_2"
    tactile_emb_type = "separate"
    img_weight = 1
    tac_weight = 10

    effective_bsz: int = 64  # 1024 # 
    device_bsz = 32  # 128 # effectie_bsz / nums_work
    native_bsz = 32  # 128 # effectie_bsz / nums_work
@dataclass
class VT20TReAll_TMR05_Bin_FT_CLS_Config(VT20TReAll_TMR05_Bin_FT_Config):
    identifier = "vt20t-reall-tmr05-bin-ft-cls"
    use_cls_token = True
@dataclass
class VT20TReAll_TMR05_Bin_FT_CLS_binary_0_55_Config(VT20TReAll_TMR05_Bin_FT_Config):
    identifier = "vt20t-reall-tmr05-bin-ft-cls-binary_0_55"
    use_cls_token = True
    tactile_type = "binary_0_55"
@dataclass
class VT20TReAll_TMR05_Bin_FT_CLS_binary_0_75_Config(VT20TReAll_TMR05_Bin_FT_Config):
    identifier = "vt20t-reall-tmr05-bin-ft-cls-binary_0_75"
    use_cls_token = True
    tactile_type = "binary_0_75"
@dataclass
class VT20TReAll_TMR05_Bin_FT_CLS_Config(VT20TReAll_TMR05_Bin_FT_Config):
    identifier = "vt20t-reall-tmr05-bin-ft-cls"
    use_cls_token = True
@dataclass
class VT20TReAll_TMR05_Bin_FT_CLS_MASKV_Config(VT20TReAll_TMR05_Bin_FT_Config):
    identifier = "vt20t-reall-tmr05-bin-ft-cls-maskv"
    use_cls_token = True
    tactile_set_to_ZERO = True
    # effective_bsz: int = 128  # 1024 # 
    # device_bsz = 64  # 128 # effectie_bsz / nums_work
    # native_bsz = 64  # 128 # effectie_bsz / nums_work
@dataclass
class VT20TReAll_VTM_TMR05_Bin_FT_CLS_Config(VT20TReAll_TMR05_Bin_FT_CLS_Config):
    identifier = "vt20t-reall-vtm-tmr05-bin-ft-cls"
    arch = "vt-reall-vtm"
    tac_one2many: bool = False
    matching_weight: float = 1

    effective_bsz = 128  # 1024 # 
    device_bsz = 64  # 128 # effectie_bsz / nums_work
    native_bsz = 64  # 128 # effectie_bsz / nums_work
@dataclass
class VT20TReAll_VTM_12m_TMR05_Bin_FT_CLS_Config(VT20TReAll_VTM_TMR05_Bin_FT_CLS_Config):
    identifier = "vt20t-reall-vtm-12M-tmr05-bin-ft-cls"
    arch = "vt-reall-vtm"
    tac_one2many: bool = True

@dataclass
class VT20TReAll_CL_TMR05_Bin_FT_CLS_Config(VT20TReAll_TMR05_Bin_FT_Config):
    arch: str = "vt-reall-cl"

    identifier = "vt20t-reall-cl-tmr05-bin-ft-cls"
    use_cls_token = True

    data_modality: str = "fk4_2f_v-sim"

    img_weight = 1
    tac_weight = 10
    cl_weight: float = 0.1

    effective_bsz: int = 256  # 1024 # 
    device_bsz = 128  # 128 # effectie_bsz / nums_work
    native_bsz = 128  # 128 # effectie_bsz / nums_work

@dataclass
class VT20TReAll_CL_VTSIM_TMR05_Bin_FT_CLS_Config(VT20TReAll_CL_TMR05_Bin_FT_CLS_Config):
    identifier = "vt20t-reall-cl-vtsim-tmr05-bin-ft-cls"
    data_modality = "fk4_2f_vt-sim"
    using_map: bool = False
@dataclass
class VT20TReAll_CL_VTSIM_GCN_TMR05_Bin_FT_CLS_Config(VT20TReAll_CL_TMR05_Bin_FT_CLS_Config):
    identifier = "vt20t-reall-cl-vtsim-gcn-tmr05-bin-ft-cls"
    data_modality = "fk4_2f_vt-sim"
    tactile_emb_type = "gcn"
    using_map: bool = False
@dataclass
class VT20TReAll_CL_VTSIM_GCN_MAP_TMR05_Bin_FT_CLS_Config(VT20TReAll_CL_TMR05_Bin_FT_CLS_Config):
    identifier = "vt20t-reall-cl-vtsim-gcn-map-tmr05-bin-ft-cls"
    data_modality = "fk4_2f_vt-sim"
    tactile_emb_type = "gcn"
    using_map: bool = True
    cl_weight: float = 1.0
@dataclass
class VT20TReAll_GCN_TMR05_Bin_FT_CLS_Config(VT20TReAll_TMR05_Bin_FT_CLS_Config):
    identifier = "vt20t-reall-gcn-tmr05-bin-ft-cls"
    tactile_emb_type = "gcn"
@dataclass
class VT20TReAll_MaskLoss_TMR05_Bin_FT_Config(VTTReAllConfig):
    arch: str = "vtt-reall-maskloss"
    identifier = "vt20t-reall-maskloss-tmr05-bin-ft"
    remarks = "mr075-fr-bce-norelu"

    use_cls_token = False
    mask_ratio = 0.75
    tactile_mask_ratio = 0.5

    tactile_set_to_ZERO = False
    tactile_type = "binary_0_2"
    tactile_emb_type = "separate"
    img_weight = 10
    tac_weight = 1
@dataclass
class VT20TReAll_MaskLoss_LOSS1_10_TMR05_Bin_FT_Config(VTTReAllConfig):
    arch: str = "vtt-reall-maskloss"
    identifier = "vt20t-reall-maskloss-loss1-10-tmr05-bin-ft-bs128-32"
    remarks = "mr075-fr-bce-norelu"

    use_cls_token = False
    mask_ratio = 0.75
    tactile_mask_ratio = 0.5

    tactile_set_to_ZERO = False
    tactile_type = "binary_0_2"
    tactile_emb_type = "separate"
    img_weight = 1
    tac_weight = 10
@dataclass
class VT20TReAll_TMR05_Bin_NoPE_FT_Config(VTTReAllConfig):
    identifier = "vt20t-reall-tmr05-bin-nope-ft"
    remarks = "mr075-fr-bce-norelu"

    use_cls_token = False
    mask_ratio = 0.75
    tactile_mask_ratio = 0.5

    tactile_set_to_ZERO = False
    tactile_type = "binary_0_2"
    tactile_emb_type = "separate_nope"
    img_weight = 1
    tac_weight = 100

@dataclass
class VT20TReAll_TCL_TMR05_Bin_FT_CLS_Config(VT20TReAll_TMR05_Bin_FT_Config):
    arch: str = "vt-reall-tcl"

    identifier = "vt20t-reall-tcl-all-patch-tmr05-bin-ft-cls"
    use_cls_token = True

    data_modality: str = "before+3in+final"

    img_weight = 1
    tac_weight = 10
    tcn_weight: float = 1
    l1_weight: float = 1e-5
    l2_weight:float = 1e-5

    n_negatives: int = 3

    eps: float = 1e-8
    effective_bsz: int = 256
    device_bsz = 64
    native_bsz = 64

    using_map: bool = False
    tac_max_pooling: bool = False
    double_encoder: bool = False
@dataclass
class VT20TReAll_TCL_GCN_TMR05_Bin_FT_CLS_Config(VT20TReAll_TCL_TMR05_Bin_FT_CLS_Config):
    identifier = "vt20t-reall-tcl-patch-gcn-tmr05-bin-ft-cls"
    tactile_emb_type = "gcn"
@dataclass
class VT20TReAll_TCL_DEN_TMR05_Bin_FT_CLS_Config(VT20TReAll_TCL_TMR05_Bin_FT_CLS_Config):
    identifier = "vt20t-reall-tcl-den-patch-gcn-tmr05-bin-ft-cls"
    effective_bsz = 128
    device_bsz = 32
    native_bsz = 32
    double_encoder = True
@dataclass
class VT20TReAll_TCL_GCN_TMR05_Bin_FT_Config(VT20TReAll_TCL_TMR05_Bin_FT_CLS_Config):
    identifier = "vt20t-reall-tcl-gcn-tmr05-bin-ft"
    tactile_emb_type = "gcn"
    use_cls_token = False
@dataclass
class VT20TReAll_TCL_TMR05_Bin_FT_Config(VT20TReAll_TCL_TMR05_Bin_FT_CLS_Config):
    identifier = "vt20t-reall-tcl-tmr05-bin-ft"
    # tactile_emb_type = "gcn"
    use_cls_token = False
@dataclass
class VT20TReAll_TCL_TMR05_Bin_FT_CLS_MAXPOOLING_Config(VT20TReAll_TCL_TMR05_Bin_FT_CLS_Config):
    identifier = "vt20t-reall-tcl-tmr05-bin-ft-cls-tacmaxpooling"
    # tactile_emb_type = "gcn"
    tac_max_pooling = True


@dataclass
class VT20TReTac_tmr05_Bin_FT_Config(VTTReTacConfig):
    identifier = "vt20t-retac-tmr05-bin-ft"
    remarks = "mr075-fr-mse"

    use_cls_token = False
    mask_ratio = 0.75
    tactile_mask_ratio = 0.5

    tactile_type = "binary_0_2"
    tactile_emb_type = "separate"
    effective_bsz: int = 64  # 1024 # 
    device_bsz = 32  # 128 # effectie_bsz / nums_work
    native_bsz = 32  # 128 # effectie_bsz / nums_work
@dataclass
class VT20TReTac_tmr05_Bin_FT_CLS_Config(VT20TReTac_tmr05_Bin_FT_Config):
    identifier = "vt20t-retac-tmr05-bin-ft-cls"
    use_cls_token = True


@dataclass
class VT20TRePic_tmr05_Bin_FT_Config(VTTRePicConfig):
    identifier = "vt20t-repic-tmr05-bin-ft"
    remarks = "mr075-fr-mse"

    use_cls_token = False
    mask_ratio = 0.75
    tactile_mask_ratio = 0.5

    tactile_set_to_ZERO = False
    tactile_type = "binary_0_2"
    tactile_emb_type = "separate"

    effective_bsz: int = 64  # 1024 # 
    device_bsz = 32  # 128 # effectie_bsz / nums_work
    native_bsz = 32  # 128 # effectie_bsz / nums_work
@dataclass
class VT20TRePic_tmr05_Bin_FT_CLS_Config(VT20TRePic_tmr05_Bin_FT_Config):
    identifier = "vt20t-repic-tmr05-bin-ft"
    use_cls_token = True

@dataclass
class T20ReTac_TMR05_Bin_FT_Config(TReTacConfig):
    identifier = "t20-retac-tmr05-bin-ft"
    remarks = "mse"

    use_cls_token = False
    tactile_mask_ratio = 0.5

    tactile_type = "binary_0_2"
    tactile_emb_type = "separate"
    effective_bsz: int = 64  # 1024 # 
    device_bsz = 32  # 128 # effectie_bsz / nums_work
    native_bsz = 32  # 128 # effectie_bsz / nums_work
@dataclass
class T20ReTac_TMR05_Bin_FT_CLS_Config(T20ReTac_TMR05_Bin_FT_Config):
    identifier = "t20-retac-tmr05-bin-ft-cls"
    use_cls_token = True
@dataclass
class T20ReTac_TMR05_Bin_FT_CLS_binary_0_55_Config(T20ReTac_TMR05_Bin_FT_Config):
    identifier = "t20-retac-tmr05-bin-ft-cls-binary_0_55"
    use_cls_token = True
    tactile_type = "binary_0_55"
@dataclass
class T20ReTac_TMR05_Bin_FT_CLS_binary_0_75_Config(T20ReTac_TMR05_Bin_FT_Config):
    identifier = "t20-retac-tmr05-bin-ft-cls-binary_0_75"
    use_cls_token = True
    tactile_type = "binary_0_75"
@dataclass
class VRePic_Bin_FT_Config(VRePicConfig):
    identifier = "v-repic-bin-ft"
    remarks = "mr075-fr-mse"

    use_cls_token = False
    mask_ratio = 0.75

    tactile_type = "binary_0_2"

    effective_bsz: int = 64  # 1024 # 
    device_bsz = 32  # 128 # effectie_bsz / nums_work
    native_bsz = 32  # 128 # effectie_bsz / nums_work

class VRePic_Bin_FT_CLS_Config(VRePic_Bin_FT_Config):
    identifier = "v-repic-bin-ft-cls"
    use_cls_token = True

@dataclass
class VRePic_CL_TMR05_Bin_FT_CLS_Config(VRePic_Bin_FT_Config):
    arch: str = "v-repic-cl"

    identifier = "v-repic-cl-tmr05-bin-ft-cls"
    use_cls_token = True

    data_modality: str = "fk4_2f_v-sim"

    img_weight: float = 1
    # tac_weight = 10
    cl_weight: float = 0.1

    effective_bsz: int = 256  # 1024 # 
    device_bsz = 128  # 128 # effectie_bsz / nums_work
    native_bsz = 128  # 128 # effectie_bsz / nums_work

@dataclass
class VR3M_NOLANG_Config(R3MSmallConfig):
    identifier = "vr3m-vit-nolang"
    arch = "vr3m-nolang"
    data_modality = "before+3in+final"
    remarks = "hahaha"

    eps: float = 1e-8
    effective_bsz: int = 1024

    device_bsz = 32
    native_bsz = 128

    tactile_type:str = "binary_0_2"
    tactile_emb_type:str = "separate"

@dataclass
class VR3M_NOLANG_CLS_Config(VR3M_NOLANG_Config):
    identifier = "vr3m-nolang-vit-cls"
    use_cls_token: bool = True

cs = ConfigStore.instance()

# recons image and tactile
cs.store(group="model", name="vt20t-reall-tmr05-bin-ft", node=VT20TReAll_TMR05_Bin_FT_Config)
cs.store(group="model", name="vt20t-reall-tmr05-bin-ft-cls", node=VT20TReAll_TMR05_Bin_FT_CLS_Config)
cs.store(group="model", name="vt20t-reall-gcn-tmr05-bin-ft-cls", node=VT20TReAll_GCN_TMR05_Bin_FT_CLS_Config)
cs.store(group="model", name="vt20t-reall-maskloss-tmr05-bin-ft", node=VT20TReAll_MaskLoss_TMR05_Bin_FT_Config)
cs.store(group="model", name="vt20t-reall-maskloss-loss1-10-tmr05-bin-ft", node=VT20TReAll_MaskLoss_LOSS1_10_TMR05_Bin_FT_Config)
cs.store(group="model", name="vt20t-reall-tmr05-bin-nope-ft", node=VT20TReAll_TMR05_Bin_NoPE_FT_Config)
cs.store(group="model", name="vt20t-reall-tmr05-bin-ft-cls-maskv", node=VT20TReAll_TMR05_Bin_FT_CLS_MASKV_Config)
cs.store(group="model", name="vt20t-reall-tmr05-bin-ft-cls-binary_0_55", node=VT20TReAll_TMR05_Bin_FT_CLS_binary_0_55_Config)
cs.store(group="model", name="vt20t-reall-tmr05-bin-ft-cls-binary_0_75", node=VT20TReAll_TMR05_Bin_FT_CLS_binary_0_75_Config)


# recons tactile
cs.store(group="model", name="vt20t-retac-tmr05-bin-ft", node=VT20TReTac_tmr05_Bin_FT_Config)
cs.store(group="model", name="vt20t-retac-tmr05-bin-ft-cls", node=VT20TReTac_tmr05_Bin_FT_CLS_Config)

# recons image
cs.store(group="model", name="vt20t-repic-tmr05-bin-ft", node=VT20TRePic_tmr05_Bin_FT_Config)
cs.store(group="model", name="vt20t-repic-tmr05-bin-ft-cls", node=VT20TRePic_tmr05_Bin_FT_CLS_Config)

# T_ReTac
cs.store(group="model", name="t20-retac-tmr05-bin-ft", node=T20ReTac_TMR05_Bin_FT_Config)
cs.store(group="model", name="t20-retac-tmr05-bin-ft-cls", node=T20ReTac_TMR05_Bin_FT_CLS_Config)
cs.store(group="model", name="t20-retac-tmr05-bin-ft-cls-binary_0_55", node=T20ReTac_TMR05_Bin_FT_CLS_binary_0_55_Config)
cs.store(group="model", name="t20-retac-tmr05-bin-ft-cls-binary_0_75", node=T20ReTac_TMR05_Bin_FT_CLS_binary_0_75_Config)

# V_RePic
cs.store(group="model", name="v-repic-bin-ft", node=VRePic_Bin_FT_Config)
cs.store(group="model", name="v-repic-bin-ft-cls", node=VRePic_Bin_FT_CLS_Config)

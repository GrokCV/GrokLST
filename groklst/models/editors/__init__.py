# GDSR
from .mocolsk import MoCoLSKNet
from .mocolsk_abs import MoCoLSKNetABS
from .mocolsk_sar import OpticalSARMoCoLSKNet
from .dyfexnet import DyFeXNet
from .dyfexnet_mask import DyFeXNetMask
from .dyfexnet_interlace import DyFeXNetInterlace
from .dct import DCTNet  # 2022
from .dkn import DKN, FDKN  # 2021
from .fdsr import FDSR  # 2021
from .p2p import PixTransformNet  # 2019
from .ahmf import AHMF  # 2022
from .djfr import DJFR  # 2019
from .dsrn import DSRN  # CVPR-2021
from .djf import DJF  # 2019
from .cunet import CUNet  # 2020
from .suft import SUFTNet  # 2022

# new gdsr
from .svlrm import SVLRM  # 2019 loss nan
from .sgnet import SGNet  # 2024 cuda out of memary
from .pmba import PMBANet2, PMBANet4, PMBANet8  # 2020 loss nan
from .rsag import RSAG  # 2023 AAAI
from .pac import PAC  # bug
from .dagf import DAGF  # TNNLS-2023 big loss
from .dgf import DGF  # CVPR-2018 big loss
from .dmsg import DMSG  # ECCV-2016 big loss
from .codon import CODONNet8, CODONNet4  # IJCV-2022


# SISR
from .srcnn import SRCNN
from .rdn import RDN
from .edsr import EDSR
from .swinir import SwinIR

# new sisr
from .act import ACT
from .ngswin import NGswin
from .omni import OmniSR
from .safmn import SAFMN
from .srfbn import SRFBN

from .fenet import FeNet
from .egasr import EGASR
from .ctnet import CTNET
from .san import SAN
from .han import HAN
from .rcan import RCAN

from .cfgn import CFGN
from .FEN import FENet
from .haunet import HAUNet
from .dctlsa import DCTLSA

from .dbpn import DBPN

__all__ = [
    # GDSR
    "MoCoLSKNet",
    "OpticalSARMoCoLSKNet",
    "MoCoLSKNetABS",
    "DyFeXNet",
    "DyFeXNetMask",
    "DyFeXNetInterlace",
    "SUFTNet",
    "DCTNet",
    "DKN",
    "FDKN",
    "FDSR",
    "PixTransformNet",
    "AHMF",
    "DJFR",
    "DSRN",
    "DJF",
    "CUNet",
    "SVLRM",
    "SGNet",
    "PMBANet2",
    "PMBANet4",
    "PMBANet8",
    "RSAG",
    "PAC",
    "DAGF",
    "DGF",
    "DMSG",
    "CODONNet4",
    "CODONNet8",
    # SISR
    "SRCNN",  # TPAMI-2015
    "RDN",  # CVPR-2018
    "EDSR",  # CVPR-2017
    "SwinIR",  # ICCVW-2021
    # new model
    "ACT",  # WACV 2023
    "NGswin",  # CVPR 2023
    "OmniSR",  # CVPR 2023
    "SAFMN",  # CVPR 2023
    "SRFBN",  # ICCV 2023
    "FeNet",
    "FENet",
    "EGASR",
    "CTNET",
    "SAN",
    "HAN",
    "RCAN",
    "CFGN",
    "FEN",
    "HAUNet",
    "DCTLSA",
    "DBPN",
]

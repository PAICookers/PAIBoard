import numpy as np
from paicorelib.framelib import FRAME_DTYPE

# 64'hffff_ffff_ffff_ffff
FRAME_PAD_DUMMY_VALUE = np.iinfo(FRAME_DTYPE).max


# Filter the frames with zero value or `FRAME_PAD_DUMMY_VALUE`
FRAME_VALUE_FILTER_LIST = [0, FRAME_PAD_DUMMY_VALUE]

DEFAULT_N_OUTPUT_FRAMES = 10_0000

SUPPORT_TOOLCHAIN = ["PAIBox"]
# Default filenames for toolchain-related files
DEFAULT_FNAME_CONFIG_FILE_WO_SUFFIX = "config_all"
DEFAULT_FNAME_GRAPH_INFO = "graph_info.json"
DEFAULT_FNAME_INPUT_NODE_INFO = "input_proj_info.json"
DEFAULT_FNAME_OUTPUT_DEST_INFO = "output_dest_info.json"
DEFAULT_FNAME_CORE_PARAMS_CONF = "core_params.json"
DEFAULT_FNAME_NEURON_PHY_LOC = "neuron_phy_loc.json"
DEFAULT_FNAME_LEARNING_MODE_DIS_CFG_FILE_WO_SUFFIX = "config_learn_dis_all"
DEFAULT_FNAME_LEARNING_MODE_EN_CFG_FILE_WO_SUFFIX = "config_learn_en_all"

CONFIG_FILE_DTYPE = "<u8"  # Little endian, 8 bytes(uint64)

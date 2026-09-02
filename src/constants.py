"project constants"

import torch

SEED = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_DATA_DIR ="data/imagewoof2"
NUM_CLASSES = 10

VIT_MEAN = [0.5, 0.5, 0.5] # print(m.default_cfg["mean"],m.default_cfg["std"])
VIT_STD = [0.5, 0.5, 0.5]
IMG_SIZE = 224  # required input size for vit_tiny_patch16_224

MODEL_NAME = "vit_tiny_patch16_224"

LORA_RANK = 10
LORA_ALPHA = None
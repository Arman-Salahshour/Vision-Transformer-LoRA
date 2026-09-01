import torch
import torch.nn as nn
SEED = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
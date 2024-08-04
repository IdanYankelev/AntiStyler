import os
import torch
from lightning_fabric.utilities.seed import seed_everything


CONTENT_LAYERS = ['conv_4']
STYLE_LAYERS = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
IMSIZE = (600, 600)
PADDING = 10
STYLE_WEIGHT = 1000
CONTENT_WEIGHT = 1
TOP_PERCENTILE = 0.99
OPTIMIZATION_STEPS = 1

def set_all_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(seed)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

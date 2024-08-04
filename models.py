from torchvision.models import vgg19
from antistyle import antistyle
from config import *

device = get_device()
style_transfer_backbone = vgg19(weights='VGG19_Weights.IMAGENET1K_V1').features.to(device).eval()
antistyle_model = antistyle(style_transfer_backbone, CONTENT_LAYERS, STYLE_LAYERS, CONTENT_WEIGHT, STYLE_WEIGHT, OPTIMIZATION_STEPS)
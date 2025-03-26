import glob

import torch
from torch import nn
from torchvision import models

from main_pytorch import train, get_default_model
from utils import GIT_ROOT


if __name__ == "__main__":
    model = get_default_model()

    MODEL_PATH = f"{GIT_ROOT}/models/VGG_0.1388118660284413"
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.features.named_parameters():
        if "28" in name:
            param.requires_grad = True

    print(model)
    train(model)

from torchvision import models
from torch import nn

from medmnist import BloodMNIST
from utils import GIT_ROOT


if __name__ == '__main__':
    DATASETS_PATH = f"{GIT_ROOT}/datasets/"

    train_data = BloodMNIST(
        "train",
        size=224,
        root=DATASETS_PATH,
        as_rgb=True
    )

    valid_data = BloodMNIST(
        "val",
        size=224,
        root=DATASETS_PATH,
        as_rgb=True
    )

    test_data = BloodMNIST(
        "test",
        size=224,
        root=DATASETS_PATH,
        as_rgb=True
    )

    model = models.vgg16(weights='DEFAULT')
    for name, param in model.named_parameters():
        if not "classifier" in name:
            param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 4096),
        nn.ReLU(),
        nn.Linear(4096, 256),
        nn.ReLU(),
        nn.Linear(256, 8)
    )


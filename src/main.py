from medmnist import BloodMNIST
from utils import GIT_ROOT


BloodMNIST(
    "train",
    download=True,
    size=128,
    root=GIT_ROOT
)

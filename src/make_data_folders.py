import os

from medmnist import BloodMNIST
from utils import GIT_ROOT


if __name__ == '__main__':
    DATASETS_PATH = f"{GIT_ROOT}/datasets/"

    if not os.path.exists(DATASETS_PATH):
        os.makedirs(DATASETS_PATH)

    train_data = BloodMNIST(
        "train",
        download=True,
        size=224,
        root=DATASETS_PATH,
        as_rgb=True
    )

    tfds_path = f"{GIT_ROOT}/datasets/tfds"
    train_path = f"{tfds_path}/train"

    print(f"Saving train set at {train_path}")
    labels_cnt = {}
    for image, label in train_data:
        label = label[0]
        if not label in labels_cnt:
            labels_cnt[label] = 0
        else:
            labels_cnt[label] += 1
        label_path = f"{train_path}/{label}"
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        image.save(f"{label_path}/{label}_{labels_cnt[label]}.png", "PNG")

    valid_data = BloodMNIST(
        "val",
        download=True,
        size=224,
        root=DATASETS_PATH,
        as_rgb=True
    )

    val_path = f"{tfds_path}/validation"
    print(f"Saving validation set at {train_path}")
    labels_cnt = {}
    for image, label in valid_data:
        label = label[0]
        if not label in labels_cnt:
            labels_cnt[label] = 0
        else:
            labels_cnt[label] += 1
        label_path = f"{val_path}/{label}"
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        image.save(f"{label_path}/{label}_{labels_cnt[label]}.png", "PNG")

    test_data = BloodMNIST(
        "test",
        download=True,
        size=224,
        root=DATASETS_PATH,
        as_rgb=True
    )
    test_data.save(
        f"{tfds_path}/test",
        write_csv=False
    )

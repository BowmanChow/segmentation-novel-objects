import random
from typing import Any, Callable, Dict, List
import numpy as np
import PIL.Image as Im
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as transformF
import os
import torch


class augmentation:
    transforms = [
    ]


IMAGE_EXTENSION = '_smoke.png'
TRUTH_EXTENSION = '_smoke_heatmap.npy'


class SegmentImageDataset(datasets.VisionDataset):
    def __init__(
        self,
        folder: str,
        image_ids: List[int],
        transforms=transforms.Compose([transforms.ToTensor()]),
    ) -> None:
        self.folder = folder
        self.image_ids = image_ids
        self.transforms = transforms
        print(f"Initialize {type(self).__name__}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img = Im.open(
            os.path.join(
                self.folder, f'{self.image_ids[index]}{IMAGE_EXTENSION}')
        )
        label = np.load(
            os.path.join(
                self.folder, f'{self.image_ids[index]}{TRUTH_EXTENSION}')
        )
        label = torch.from_numpy(label)
        label = label.unsqueeze(0).unsqueeze(0)
        label = F.interpolate(
            label,
            size=(
                img.size[0], img.size[1]), mode='bicubic')
        label = label.squeeze(0)

        if self.transforms:
            img = self.transforms(img)

        return img, label

    def save_torch(self, index, path: str, suffix: str):
        img_torch, label_torch = self.__getitem__(index)
        img_np = img_torch.numpy()
        label_np = label_torch.numpy()
        if len(img_np.shape) == 3:
            img_np = np.moveaxis(img_np, 0, 2)
        img = Im.fromarray((255.0 * img_np).astype(np.uint8))
        img.save(os.path.join(path, f"imaging_{suffix}.png"))
        label = Im.fromarray((127.0 * label_np).astype(np.uint8))
        label.save(os.path.join(path, f"label_{suffix}.png"))


def split_ids(cases: list, split_num):
    random_cases = random.sample(cases, len(cases))
    list1 = random_cases[:split_num]
    list1.sort()
    list2 = random_cases[split_num:]
    list2.sort()
    return list1, list2


def data_loader(
        folder: str,
        train_image_ids: list, valid_image_ids: list, train_batch_size: int, valid_batch_size: int, is_normalize: bool = True):
    print(f"""Data loader:
{train_image_ids = }
{valid_image_ids = }
""")

    train_data = SegmentImageDataset(
        folder=folder,
        image_ids=train_image_ids,
    )
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=6,
    )

    val_data = SegmentImageDataset(
        folder=folder,
        image_ids=valid_image_ids,
    )
    valid_loader = DataLoader(
        dataset=val_data,
        batch_size=valid_batch_size,
        num_workers=6,
    )

    return train_loader, valid_loader


if __name__ == '__main__':
    # cases_to_numpy()
    from matplotlib import pyplot as plt
    dataset = SegmentImageDataset(
        '../stablediffusion/outputs/img_2_img_heatmap', list(range(460, 1000)))
    print(f"{dataset.__len__()}")
    # dataset.save_np(100)
    img, label = dataset[300]
    print(img)
    print(img.shape)
    print(label)
    print(label.shape)
    save_image(img, "image.png")
    plt.imshow(label.numpy())
    plt.colorbar()
    plt.savefig('test.png')

"""Dataloader"""

import os
import random
import math
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


class CityScapesDataset(Dataset):
    def __init__(self, args, purpose, h=2048, w=1024) -> None:
        self.args = args
        self.purpose = purpose
        self.crop_size = args.crop_size
        self.label_dict = args.label_dict
        self.palette = args.palette
        self.model = args.model

        self.labels_list = []
        self.image_list = []

        self.labels_list = glob.glob(
            os.path.join(args.gtFine_dir, purpose, "*/*_gtFine_labelIds.png")
        )
        self.image_list = glob.glob(
            os.path.join(args.image_dir, purpose, "*/*_leftImg8bit.png")
        )

        self.labels_list.sort()
        # self.instance_list.sort()
        self.image_list.sort()
        self.transform = self.get_transform(h, w)

    def get_transform(self, h, w):
        h, w = self.short_side(h, w, self.crop_size)

        return transforms.Compose(
            [
                transforms.Resize([h, w], Image.NEAREST),
                transforms.RandomCrop((self.crop_size, self.crop_size * 2)),
            ]
        )

    def short_side(self, w, h, size):
        # https://github.com/facebookresearch/pytorchvideo/blob/a77729992bcf1e43bf5fa507c8dc4517b3d7bc4c/pytorchvideo/transforms/functional.py#L118
        if w < h:
            new_h = int(math.floor((float(h) / w) * size))
            new_w = size
        else:
            new_h = size
            new_w = int(math.floor((float(w) / h) * size))
        return new_w, new_h

    def get_img(self, image_file_path):
        pil_image = Image.open(image_file_path)

        if self.model == "Mask2Former":
            return pil_image
        else:
            image_numpy = np.array(pil_image)
            image_tensor = torch.from_numpy(image_numpy).permute(2, 0, 1)
            return image_tensor

    def get_label(self, labels_file_path):
        pil_labels = Image.open(labels_file_path)
        labels_numpy = np.array(pil_labels)

        # labelIDをtrainlabelIDに変換
        labels_numpy = np.vectorize(self.label_dict.get)(labels_numpy)
        if self.model == "Mask2Former":
            # 19->255
            labels_numpy[labels_numpy == 19] = 255

        return torch.from_numpy(labels_numpy).unsqueeze(0)

    def get_one_pair_unet(self, index, labels_list, image_list) -> dict:
        labels_file_path = labels_list[index]
        image_file_path = image_list[index]

        image_tensor = self.get_img(image_file_path)
        labels_tensor = self.get_label(labels_file_path)

        if self.model == "Unet":
            seed = random.randint(0, 2**32)
            torch.manual_seed(seed)
            image_tensor = self.transform(image_tensor)
            torch.manual_seed(seed)
            labels_tensor = self.transform(labels_tensor)

        return {
            "labels": labels_tensor,
            # 'instance': instance_tensor,
            "image": image_tensor,
        }

    def get_one_pair_mask2former(self, index, labels_list, image_list) -> dict:
        labels_file_path = labels_list[index]
        image_file_path = image_list[index]

        image_pillow = self.get_img(image_file_path)
        labels_tensor = self.get_label(labels_file_path)

    def __getitem__(self, index) -> dict:
        data = self.get_one_pair_unet(
            index,
            self.labels_list,
            # self.instance_list,
            self.image_list,
        )
        # data = self.get_one_pair_mask2former()

        return data

    def __len__(self):
        return len(self.labels_list)


def dataset_facory(args):
    train_dataset = CityScapesDataset(args, purpose="train")
    val_dataset = CityScapesDataset(args, purpose="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, val_loader

import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy
import random
import math
import glob


class AlignedDataset(Dataset):
    def __init__(self, args, purpose) -> None:
        # データセットクラスの初期化
        self.args = args
        self.purpose = purpose
        self.crop_size = args.crop_size
        self.label_dict = args.label_dict
        self.palette = args.palette
        self.model = args.model

        # labelsファイルのパスのリスト
        self.labels_list = []
        # imageファイルのパスのリスト
        self.image_list = []

        self.labels_list = glob.glob(
            os.path.join(args.gtFine_dir, purpose, "*/*_gtFine_labelIds.png")
        )
        self.image_list = glob.glob(
            os.path.join(args.image_dir, purpose, "*/*_leftImg8bit.png")
        )

        # ソートする
        self.labels_list.sort()
        # self.instance_list.sort()
        self.image_list.sort()
        # これでlabels，instance，imageのそれぞれの画像までのパス一覧が順番に並んだリストができた

    # 短い方をsizeの値に合わせるように，アスペクト比を保ったままリサイズする
    def short_side(self, w, h, size):
        # https://github.com/facebookresearch/pytorchvideo/blob/a77729992bcf1e43bf5fa507c8dc4517b3d7bc4c/pytorchvideo/transforms/functional.py#L118
        if w < h:
            new_h = int(math.floor((float(h) / w) * size))
            new_w = size
        else:
            new_h = size
            new_w = int(math.floor((float(w) / h) * size))
        return new_w, new_h

    def make_dataset(self, index, labels_list, image_list) -> dict:
        # ランダムなindexの画像を取得
        labels_file_path = labels_list[index]
        # instance_file_path = instance_list[index]
        image_file_path = image_list[index]

        # seed = random.randint(0, 2**32)
        seed = 0

        # shape:H*W*3が欲しい
        # ----実画像-----
        # img->pil
        pil_image = Image.open(image_file_path)

        if self.model == "Mask2Former":
            # 画像とラベルの前処理をプリトレーニド処理器で行う
            # image_processor = self.processor(images=pil_image, return_tensors="pt")
            # ここまでの前処理はOK
            # image_tensor["pixel_values"] = image_tensor["pixel_values"].squeeze(0)
            # image_tensor["pixel_mask"] = image_tensor["pixel_mask"].squeeze(0)
            image_tensor = image_file_path
        else:
            # pil->np
            image_numpy = numpy.array(pil_image)
            # np->tensor
            image_tensor = torch.from_numpy(image_numpy).permute(2, 0, 1)
            # shape：[3, 1024, 2048]

        # shape:H*W欲しい
        # ----ラベル画像----
        # img->pil
        pil_labels = Image.open(labels_file_path)
        # print(labels_file_path)

        # pil->np
        labels_numpy = numpy.array(pil_labels)

        # labelIDをtrainlabelIDに変換
        labels_numpy = numpy.vectorize(self.label_dict.get)(labels_numpy)

        # np->tensor
        labels_tensor = torch.from_numpy(labels_numpy).unsqueeze(0)

        h, w = self.short_side(
            labels_tensor.size()[0], labels_tensor.size()[1], self.crop_size
        )

        transform_list = [
            transforms.Resize([h, w], Image.NEAREST),
            transforms.RandomCrop((self.crop_size, self.crop_size * 2)),
        ]

        if self.model == "Mask2Former":
            labels_tensor = labels_tensor.squeeze(0)

        # transform.Compose：複数のTransformを連続して行うTransform
        transform = transforms.Compose(transform_list)

        if self.model == "Unet":
            torch.manual_seed(seed)
            image_tensor = transform(image_tensor.float())

        torch.manual_seed(seed)
        labels_tensor = transform(labels_tensor.float())

        return {
            "labels": labels_tensor,
            # 'instance': instance_tensor,
            "image": image_tensor,
        }

    def __getitem__(self, index) -> dict:
        data = self.make_dataset(
            index,
            self.labels_list,
            # self.instance_list,
            self.image_list,
        )

        return data

    def __len__(self):
        # 全画像ファイル数を返す
        return len(self.labels_list)


def dataset_facory(args):
    train_dataset = AlignedDataset(args, purpose="train")
    val_dataset = AlignedDataset(args, purpose="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,  # argsで指定
        shuffle=True,  # ランダムあり
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,  # argsで指定
        shuffle=False,  # ランダムなし
        pin_memory=True,
    )

    return train_loader, val_loader
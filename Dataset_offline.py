import os.path
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy
import random
import math
import glob
from pathlib import Path
from skimage.io import imread

# バッチ単位で乱数をふる


class AlignedDataset(Dataset):
    def __init__(self, args, purpose) -> None:
        # データセットクラスの初期化
        self.args = args
        self.purpose = purpose
        self.crop_size = args.crop_size
        self.label_dict = args.label_dict
        self.palette = args.palette

        # labelsファイルのパスのリスト
        self.labels_list = []
        # instanceファイルのパスのリスト
        # self.instance_list = []
        # imageファイルのパスのリスト
        self.image_list = []

        # purpose == trainの時，trainのパスを指定
        if purpose == "train":
            self.labels_list = glob.glob(
                os.path.join(args.gtFine_dir, "train/*/*_gtFine_labelIds.png")
            )
            # self.instance_list = glob.glob(args.gtFine_dir + 'train/*/*_gtFine_instanceIds.png')
            self.image_list = glob.glob(
                os.path.join(args.image_dir, "train/*/*_leftImg8bit.png")
            )
        # purpose == valの時，valのパスを指定
        elif purpose == "val":
            self.labels_list = glob.glob(
                os.path.join(args.gtFine_dir, "val/*/*_gtFine_labelIds.png")
            )
            # self.instance_list = glob.glob(args.gtFine_dir + 'val/*/*_gtFine_instanceIds.png')
            self.image_list = glob.glob(
                os.path.join(args.image_dir, "val/*/*_leftImg8bit.png")
            )
        # purpose == testの時，testのパスを指定
        else:
            self.labels_list = glob.glob(
                os.path.join(args.gtFine_dir, "test/*/*_gtFine_labelIds.png")
            )
            # self.instance_list = glob.glob(os.path.join(args.gtFine_dir, 'test/*/*_gtFine_instanceIds.png'))
            self.image_list = glob.glob(
                os.path.join(args.image_dir, "test/*/*_leftImg8bit.png")
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

        seed = random.randint(0, 2**32)
        # for in range(開始値，最大値，間隔)
        # for i in range(
        #     rand_index,
        #     rand_index + len(image_file_path) - rand_index,
        #     1
        # ):

        # shape:H*W*3が欲しい
        # ----実画像-----
        # img->pil
        pil_image = Image.open(image_file_path)
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

        # # shape:H*Wが欲しい
        # # ----インスタンスマップ----
        # # img->pil
        # pil_instance = Image.open(instance_file_path)
        # # pil->np
        # instance_numpy = numpy.array(pil_instance)
        # # np->tensor
        # instance_tensor = torch.from_numpy(instance_numpy).unsqueeze(0)

        # リサイズされた画像の縦と横の長さをリスト化する
        # image.size()[1]：縦の長さ，image.size()[2]：横の長さ
        h, w = self.short_side(
            image_tensor.size()[1], image_tensor.size()[2], self.crop_size
        )
        transform_list = [
            transforms.Resize([h, w], Image.NEAREST),
            transforms.RandomCrop((self.args.crop_size, self.args.crop_size * 2)),
        ]

        # transform.Compose：複数のTransformを連続して行うTransform
        transform = transforms.Compose(transform_list)

        torch.manual_seed(seed)
        image_tensor = transform(image_tensor.float())

        torch.manual_seed(seed)
        labels_tensor = transform(labels_tensor.float())

        # instance_tensor.size():16*1*224*224
        # instance_tensor = torch.stack(instance, dim=0)
        # torch.manual_seed(seed)
        # instance_tensor = transform(instance_tensor)

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
    # trainが
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

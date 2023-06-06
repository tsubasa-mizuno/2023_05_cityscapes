from util import AverageMeter
from tqdm import tqdm
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerForUniversalSegmentationOutput,
)
from transformers import AutoImageProcessor
import torch
from imagesave import imagesave
import numpy
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F


def train(
    model,
    processor,
    device,
    criterion,
    optimizer,
    loader,
    iters,
    epoch,
    experiment,
    evaluator,
    args,
    global_step,
):
    evaluator.reset()

    model.train()

    train_loss = AverageMeter()

    with tqdm(loader, leave=False) as pbar_train:
        pbar_train.set_description("[train]")

        for sample in pbar_train:
            image, labels = (
                sample["image"],  # B, C, H, W
                sample["labels"],  # B, 1, H, W
            )
            labels = labels.to(device)
            labels = labels.squeeze(dim=1)  # B, H, W
            bs = len(image)
            # labels_shape[8, 256, 512]
            if args.model == "Mask2Former":
                for i in range(bs):
                    image[i] = Image.open(image[i])
                inputs = processor(images=image, return_tensors="pt")
                inputs["pixel_values"] = inputs["pixel_values"].to(device)
                inputs["pixel_mask"] = inputs["pixel_mask"].to(device)
                # mask_labels_2 = torch.stack(
                #     [(labels == i).long() for i in range(args.num_class - 1)]
                # )
                # mask_labels_2 = mask_labels_2.long()

                # グランドトゥルースオブジェクトの数を取得
                labels = labels.long()  # B, H, W: int64
                num_target_boxes = labels.max() + 1

                class_labels = []
                mask_labels = []

                # 各オブジェクトに対して処理を行う
                for i in range(num_target_boxes):
                    # オブジェクトの領域を特定
                    mask = labels == i  # B, H, W: bool

                    # マスクが空でないことを確認してからクラスラベルを取得
                    if mask.any():
                        class_label = labels * mask  # B, H, W: int64
                    else:
                        # Handle the case when the mask is empty
                        class_label = None

                    # オブジェクトのマスクを保存
                    mask_labels.append(mask)

                    # オブジェクトのクラスラベルを保存
                    class_labels.append(class_label)

                # Noneの値をフィルタリングする
                class_labels = [label for label in class_labels if label is not None]

                # 結果をテンソルに変換
                class_labels = (
                    torch.stack(class_labels)  # list of [B, H, W] to (N, B, H, W)
                    .reshape(len(class_labels), bs, -1)  # (N, B, HW)
                    .max(dim=2)  # (N, B)
                    .values
                )
                mask_labels = torch.stack(mask_labels).float()

                outputs = model(
                    **inputs, class_labels=class_labels, mask_labels=mask_labels
                )
                loss = outputs.loss
                # target_shape[256, 512]

            else:
                image = image.to(device).float()
                target = model(image)
                loss = criterion(target, labels.long())
                # target_shape[8, 3, 256, 512]

            # imagesave(target, labels, args, 3, epoch)
            train_loss.update(loss, labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            experiment.log_metric("train_loss", loss, epoch=epoch, step=global_step)

    experiment.log_metric(
        "train_epoch_loss", train_loss.avg, epoch=epoch, step=global_step
    )
    global_step += 1

    return iters, global_step

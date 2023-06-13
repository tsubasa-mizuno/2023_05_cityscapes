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
    i = 0

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

                class_labels = []
                mask_labels = []
                for b in range(bs):
                    class_label_of_b = labels[b].unique()
                    class_labels.append(class_label_of_b)  # list of [num_labels] long?

                    # 各カテゴリiのmaskに対して処理を行う
                    mask_labels_of_b = []
                    for i in class_label_of_b:
                        mask = labels[b] == i  # H, W: bool
                        mask_labels_of_b.append(mask.float())  # list of [H, W] float
                    mask_labels_of_b = torch.stack(mask_labels_of_b)

                    mask_labels.append(
                        mask_labels_of_b
                    )  # list of [num_labels, H, W] float

                outputs = model(
                    **inputs, class_labels=class_labels, mask_labels=mask_labels
                )
                # target = processor.post_process_semantic_segmentation(
                #     outputs, target_sizes=[image[0].size[::-1]]
                # )[0]

                loss = outputs.loss
                # target_shape[256, 512]

            else:
                image = image.to(device).float()
                target = model(image)
                loss = criterion(target, labels.long())
                # target_shape[8, 3, 256, 512]

            # imagesave(target, labels, args, i, epoch)
            i += 1
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

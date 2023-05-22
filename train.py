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

    processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-small-cityscapes-semantic"
    )

    with tqdm(loader, leave=False) as pbar_train:
        pbar_train.set_description("[train]")
        for sample in pbar_train:
            image, labels = (
                sample["image"],
                sample["labels"],
            )
            labels = labels.to(device)
            labels = labels.squeeze(dim=1)
            # labels_shape[8, 256, 512]
            if args.model == "Mask2Former":
                # target = [0] * 8
                for i in range(8):
                    image[i] = Image.open(image[i])
                inputs = processor(images=image, return_tensors="pt")
                inputs["pixel_values"] = inputs["pixel_values"].to(device)
                inputs["pixel_mask"] = inputs["pixel_mask"].to(device)
                outputs = model(**inputs)
                target_sizes = [[256, 512]] * 8
                # probabilities = F.softmax(outputs, dim=1)
                # predictions = torch.argmax(probabilities, dim=1)
                target = processor.post_process_semantic_segmentation(
                    outputs, target_sizes
                )[0]
                # target_shape[256, 512]
                target = target.unsqueeze(0)
                # target = target.unsqueeze(0)
                target = target.expand(8, -1, -1)  # [3, 256, 512]にサイズを変更
                target = target.float()
                # target_shape[8, 3, 256, 512]

            else:
                image = image.to(device).float()
                target = model(image)
                # target_shape[8, 3, 256, 512]

            # imagesave(target, labels, args, 3, epoch)

            loss = criterion(target, labels.float())
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

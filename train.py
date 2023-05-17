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


def train(
    model,
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
            if args.model == "Mask2Former":
                image = Image.open(image[0])
                inputs = processor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                target = processor.post_process_semantic_segmentation(
                    outputs, [256, 512] * 8
                )[0]
                # target_shape[256, 512]
            else:
                image = image.cuda().float()
                target = model(image)
                # target_shape[8, 3, 256, 512]

            labels = labels.cuda()
            labels = labels.squeeze(dim=1)
            # labels_shape[8, 256, 512]

            imagesave(target, labels, args, 3, epoch)

            loss = criterion(target.float(), labels.long())
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

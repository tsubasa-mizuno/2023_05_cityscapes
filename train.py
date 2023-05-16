from util import AverageMeter
from tqdm import tqdm
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerForUniversalSegmentationOutput,
)
import torch.nn.functional as F


def train(
    model,
    criterion,
    optimizer,
    loader,
    iters,
    epoch,
    experiment,
    evaluator,
    global_step,
):
    evaluator.reset()

    model.train()

    train_loss = AverageMeter()

    with tqdm(loader, leave=False) as pbar_train:
        pbar_train.set_description("[train]")
        for sample in pbar_train:
            image, labels = sample["image"], sample["labels"]
            image = image.cuda().float()
            # if args.model == "Mask2Former":
            #     image = image_processor(image, return_tensors="pt")

            labels = labels.cuda()
            labels = labels.squeeze(dim=1)
            target = model(image)
            if isinstance(target, Mask2FormerForUniversalSegmentationOutput):
                target = target.masks_queries_logits

            loss = criterion(target, labels.long())
            train_loss.update(loss, image.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            experiment.log_metric("train_loss", loss, epoch=epoch, step=global_step)

    experiment.log_metric(
        "train_epoch_loss", train_loss.avg, epoch=epoch, step=global_step
    )
    global_step += 1

    return iters, global_step

from util import AverageMeter
from tqdm import tqdm
from imagesave import imagesave
import torch


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

    i = 0

    with tqdm(loader, leave=False) as pbar_train:
        pbar_train.set_description("[train]")
        for sample in pbar_train:
            image, labels = sample["image"], sample["labels"]
            image = image.cuda()
            labels = labels.cuda()
            labels = labels.squeeze(dim=1)
            target = model(image)
            loss = criterion(target, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.update(loss, image.size(0))

    experiment.log_metric(
        "train_epoch_loss", train_loss.avg, epoch=epoch, step=global_step
    )
    return iters, global_step

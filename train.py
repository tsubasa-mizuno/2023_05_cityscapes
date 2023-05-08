from util import AverageMeter
from tqdm import tqdm
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

    with tqdm(loader, leave=False) as pbar_train:
        pbar_train.set_description("[train]")
        for sample in pbar_train:
            image, labels = sample["image"], sample["labels"]
            image = image.cuda()
            labels = labels.cuda()
            labels = labels.squeeze(dim=1)
            y = model(image)
            loss = criterion(y, labels.long())
            train_loss.update(loss, image.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    experiment.log_metric(
        "train_epoch_loss", train_loss.avg, epoch=epoch, step=global_step
    )
    return iters, train_loss

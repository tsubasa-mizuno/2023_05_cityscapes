from util import AverageMeter
from tqdm import tqdm


def train(model, criterion, optimizer, loader, iters, epoch, experiment, evaluator):
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
            target = model(image)
            loss = criterion(target, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.update(loss, image.size(0))

    epoch_loss = train_loss.avg
    experiment.log_metric("epoch_loss", epoch_loss, step=epoch)
    return iters, train_loss

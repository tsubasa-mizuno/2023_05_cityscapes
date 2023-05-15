from util import AverageMeter
from tqdm import tqdm


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

    # i = 0

    with tqdm(loader, leave=False) as pbar_train:
        pbar_train.set_description("[train]")
        for sample in pbar_train:
            image, labels = sample["image"], sample["labels"]
            image = image.cuda()
            labels = labels.cuda()
            labels = labels.squeeze(dim=1)
            # if args.model == "Mask2Former":
            #     image = preprocess(image)
            target = model(image)
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

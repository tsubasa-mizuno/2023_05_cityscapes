import torch
from make_sample import make_sample
from util import save_checkpoint
from util import AverageMeter
import numpy
from imagesave import imagesave


def val(model, criterion, epoch, val_loader, evaluator, experiment, args, global_step):
    model.eval()

    val_loss = AverageMeter()
    i = 0
    c = 0

    for sample in val_loader:
        image, labels = sample["image"], sample["labels"]

        image = image.cuda()

        # labelの値の修正
        labels = labels.float().cuda()
        if torch.max(labels) >= 30:
            labels[labels >= 30] = 29
            c += 1

        labels = labels.squeeze(dim=1)

        with torch.no_grad():
            target = model(image)

        imagesave(image, target, labels, args, i)
        i += 1

        loss = criterion(target, labels.long())
        val_loss.update(loss.item(), image.size(0))
        pred = torch.argmax(target, dim=1)
        pred = pred.data.cpu().numpy()
        label1 = labels.cpu().numpy()
        evaluator.add_batch(label1, pred)

    if epoch % args.save_epochs == 0:
        make_sample(image, labels, model, experiment, epoch, dataset=args.dataset)
        save_checkpoint(
            model,
            # "{0}_{1}_checkpoint_{}.pth".format(args.model, args.dataset, epoch),
            "{}_{}_checkpoint_{}.pth".format(args.model, args.dataset, epoch),
            args.dir_data_name,
        )

    mIoU = evaluator.Mean_Intersection_over_Union()
    accuracy = evaluator.Pixel_Accuracy()

    experiment.log_metric("val_epoch_loss", val_loss.avg, epoch=epoch, step=global_step)
    experiment.log_metric("val_epoch_accuracy", accuracy, epoch=epoch, step=global_step)
    experiment.log_metric("val_epoch_mIoU", mIoU, epoch=epoch, step=global_step)

    return global_step

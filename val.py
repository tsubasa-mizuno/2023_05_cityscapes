import torch
from make_sample import make_sample
from util import save_checkpoint
from util import AverageMeter
import numpy
from imagesave import imagesave


def val(model, criterion, epoch, val_loader, evaluator, experiment, args):
    model.eval()

    val_loss = AverageMeter()
    i = 0

    for sample in val_loader:
        image, labels = sample["image"], sample["labels"]

        image = image.cuda()

        # labelの値の修正
        labels = labels.numpy().astype(numpy.float32)
        max_label_value = numpy.max(labels)
        if max_label_value >= 30:
            labels[labels >= 30] = 29
        labels = torch.from_numpy(labels)

        labels = labels.squeeze(dim=1)
        labels = labels.cuda()

        with torch.no_grad():
            target = model(image)

        imagesave(target, labels, args, i)
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

    experiment.log_metric("val_epoch_loss", val_loss.avg, step=epoch)
    experiment.log_metric("val_epoch_accuracy", accuracy, step=epoch)
    experiment.log_metric("val_epoch_mIoU", mIoU, step=epoch)

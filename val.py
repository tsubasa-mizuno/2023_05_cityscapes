import torch
from make_sample import make_sample
from util import save_checkpoint
from util import AverageMeter
from imagesave import imagesave


def val(
    model, criterion, epoch, val_loader, evaluator, experiment, args, global_step, c
):
    model.eval()

    val_loss = AverageMeter()
    i = 0

    for sample in val_loader:
        image, labels = sample["image"], sample["labels"]
        image = image.cuda()
        labels = labels.cuda()
        labels = labels.squeeze(dim=1)

        with torch.no_grad():
            target = model(image)

        imagesave(target, labels, args, i, c)
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
            "{}_{}_checkpoint_{}.pth".format(args.model, args.dataset, epoch),
            args.dir_data_name,
        )

    mIoU = evaluator.Mean_Intersection_over_Union()
    accuracy = evaluator.Pixel_Accuracy()

    # val_loss_avg = val_loss.avg
    experiment.log_metric("val_epoch_loss", val_loss.avg, epoch=epoch, step=global_step)
    experiment.log_metric("val_epoch_accuracy", accuracy, epoch=epoch, step=global_step)
    experiment.log_metric("val_epoch_mIoU", mIoU, epoch=epoch, step=global_step)

    return global_step

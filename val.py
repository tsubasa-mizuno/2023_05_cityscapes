import torch
from make_sample import make_sample
from util import save_checkpoint
from util import AverageMeter


def val(model, criterion, epoch, val_loader, evaluator, experiment, args):
    model.eval()

    loss = AverageMeter()

    for sample in val_loader:
        image, labels = sample["image"], sample["labels"]

        image = image.cuda()
        labels = labels.cuda()
        labels = labels.squeeze(1)
        labels = labels.long()

        with torch.no_grad():
            target = model(image)

        # criterion([N,C,H,W], [N,H,W])
nn        loss_val = criterion(target, labels)
        loss.update(loss_val.item(), image.size(0))

        pred = torch.argmax(target, dim=1)
        pred = pred.data.cpu().numpy()
        labels1 = labels.cpu().numpy()
        evaluator.add_batch(labels1, pred)

    if epoch % args.save_epochs == 0:
        make_sample(image, labels, model, experiment, epoch, dataset=args.dataset)
        save_checkpoint(
            model,
            "{0}_{1}_checkpoint_{2}.pth".format(args.model, args.dataset, epoch),
            args.dir_data_name,
        )

    mIoU = evaluator.Mean_Intersection_over_Union()
    accuracy = evaluator.Pixel_Accuracy()

    experiment.log_metric("val_epoch_loss", loss.avg, step=epoch)
    experiment.log_metric("epoch_accuracy", accuracy, step=epoch)
    experiment.log_metric("epoch_mIoU", mIoU, step=epoch)

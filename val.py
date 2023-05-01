import torch
from make_sample import make_sample
from util import save_checkpoint
from util import AverageMeter
from tqdm import tqdm


# def val(model, criterion, epoch, val_val_loader, evaluator, experiment, args):
#     evaluator.reset()

#     model.eval()

#     val_loss = AverageMeter()

#     with torch.no_grad():
#         for sample in val_val_loader:
#             image, labels = sample["image"], sample["labels"]
#             image = image.cuda()
#             labels = labels.cuda()
#             labels = labels.squeeze(dim=1)
#             target = model(image)
#             loss = criterion(target, labels.long())
#             val_loss.update(loss, image.size(0))

#             pred = torch.argmax(target, dim=1)
#             pred = pred.data.cpu().numpy()
#             labels1 = labels.cpu().numpy()
#             evaluator.add_batch(labels1, pred)

#     if epoch % args.save_epochs == 0:
#         make_sample(image, labels, model, experiment, epoch, dataset=args.dataset)
#         save_checkpoint(
#             model,
#             "{0}_{1}_checkpoint_{2}.pth".format(args.model, args.dataset, epoch),
#             args.dir_data_name,
#         )

#     mIoU = evaluator.Mean_Intersection_over_Union()
#     accuracy = evaluator.Pixel_Accuracy()

#     experiment.log_metric("val_epoch_loss", val_loss.avg, step=epoch)
#     experiment.log_metric("epoch_accuracy", accuracy, step=epoch)
#     experiment.log_metric("epoch_mIoU", mIoU, step=epoch)


# def val(model, criterion, epoch, loader, evaluator, experiment, args):
#     model.eval()
#     val_loss = AverageMeter()

#     with tqdm(loader, leave=False) as pbar_val:
#         pbar_val.set_description("[val]")
#         for sample in pbar_val:
#             image, labels = sample["image"], sample["labels"]
#             image = image.cuda()
#             labels = labels.cuda()
#             labels = labels.squeeze(dim=1)
#             with torch.no_grad():
#                 target = model(image)
#                 loss = criterion(target, labels.long())
#                 val_loss.update(loss, image.size(0))
#                 pred = torch.argmax(target, dim=1)
#                 evaluator.add_batch(labels.cpu().numpy(), pred.cpu().numpy())

#     mean_IoU, per_class_IoU = evaluator.Mean_Intersection_over_Union()
#     experiment.log_metric("epoch_val_loss", val_loss.avg, step=epoch)
#     experiment.log_metric("mean_IoU", mean_IoU, step=epoch)
#     experiment.log_metric("per_class_IoU", per_class_IoU, step=epoch)

#     return mean_IoU, per_class_IoU, val_loss


def val(model, criterion, epoch, val_loader, evaluator, experiment, args):
    model.eval()

    val_loss = AverageMeter()

    for sample in val_loader:
        image, labels = sample["image"], sample["labels"]

        image = image.cuda()
        labels = labels.cuda()
        labels = labels.squeeze(dim=1)
        with torch.no_grad():
            target = model(image)

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
    experiment.log_metric("epoch_accuracy", accuracy, step=epoch)
    experiment.log_metric("epoch_mIoU", mIoU, step=epoch)

import torch
from make_sample import make_sample
from util import save_checkpoint
from util import AverageMeter
from imagesave import imagesave
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerForUniversalSegmentationOutput,
)
from PIL import Image


def val(
    model,
    device,
    criterion,
    epoch,
    val_loader,
    evaluator,
    experiment,
    args,
    global_step,
):
    model.eval()

    val_loss = AverageMeter()
    i = 0

    for sample in val_loader:
        image, labels = sample["image"], sample["labels"]
        if args.model == "Mask2Former":
            target = [0] * 8
            image[i] = Image.open(image[i])
            inputs = processor(images=image[i], return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"].cuda()
            inputs["pixel_mask"] = inputs["pixel_mask"].cuda()
            outputs = model(**inputs)
            target[i] = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image[i].size[::-1]]
            )[0]

        else:
            image = image.squeeze(dim=1)
            image = image.cuda().float()

        labels = labels.cuda()
        labels = labels.squeeze(dim=1)

        with torch.no_grad():
            target = model(image)
            if isinstance(target, Mask2FormerForUniversalSegmentationOutput):
                target = target.masks_queries_logits
                target = torch.nn.functional.interpolate(
                    target,
                    size=(384, 384),
                    mode="bilinear",
                    align_corners=False,
                )

        imagesave(target, labels, args, i, epoch)
        i += 1

        loss = criterion(target, labels.long())
        val_loss.update(loss.item(), labels.size(0))
        pred = torch.argmax(target, dim=1)
        pred = pred.data.cpu().numpy()
        label1 = labels.cpu().numpy()
        evaluator.add_batch(label1, pred)

        experiment.log_metric("val_loss", loss, epoch=epoch, step=global_step)

    if epoch % args.save_epochs == 0:
        make_sample(image, labels, model, experiment, epoch, dataset=args.dataset)
        save_checkpoint(
            model,
            "{}_{}_checkpoint_{}.pth".format(args.model, args.dataset, epoch),
            args.dir_data_name,
        )

    mIoU = evaluator.Mean_Intersection_over_Union()
    accuracy = evaluator.Pixel_Accuracy()

    experiment.log_metric("val_epoch_loss", val_loss.avg, epoch=epoch, step=global_step)
    experiment.log_metric("val_epoch_accuracy", accuracy, epoch=epoch, step=global_step)
    experiment.log_metric("val_epoch_mIoU", mIoU, epoch=epoch, step=global_step)

    return global_step

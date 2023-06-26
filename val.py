"""validation function"""

import torch
from PIL import Image
from make_sample import make_sample
from util import save_checkpoint
from util import AverageMeter
from imagesave import imagesave


def val(
    model,
    processor,
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

    with torch.no_grad():
        for sample in val_loader:
            image, labels = (
                sample["image"],  # B, C, H, W
                sample["labels"],  # B, 1, H, W
            )
            labels = labels.to(device)
            labels = labels.squeeze(dim=1)  # B, H, W
            bs = len(image)

            if args.model == "Mask2Former":
                for i in range(bs):
                    image[i] = Image.open(image[i])

                inputs = processor(images=image, return_tensors="pt")
                inputs["pixel_values"] = inputs["pixel_values"].to(device)
                inputs["pixel_mask"] = inputs["pixel_mask"].to(device)

                class_labels = []
                mask_labels = []
                for b in range(bs):
                    class_label_of_b = labels[b].unique().long()
                    class_labels.append(class_label_of_b)  # list of [num_labels] long?

                    mask_labels_of_b = []
                    for i in class_label_of_b:
                        mask = labels[b] == i  # H, W: bool
                        mask_labels_of_b.append(mask.float())  # list of [H, W] float
                    mask_labels_of_b = torch.stack(mask_labels_of_b)

                    mask_labels.append(
                        mask_labels_of_b
                    )  # list of [num_labels, H, W] float

                outputs = model(
                    **inputs, class_labels=class_labels, mask_labels=mask_labels
                )

                target = []
                original_size = image[0].size[::-1]
                quarter_size = (original_size[0] // 4, original_size[1] // 4)

                target = processor.post_process_semantic_segmentation(
                    outputs, target_sizes=([quarter_size, quarter_size])
                )
                for b in range(bs):
                    target_img = target[b]
                    imagesave(target_img, labels, args, i, epoch, bs)

                target = torch.stack(target)
                pred = target
                loss = outputs.loss

            else:
                image = image.to(device).float()
                target = model(image)
                pred = torch.argmax(target, dim=1)
                loss = criterion(target, labels.long())
                bs = 1
                imagesave(target, labels, args, i, epoch, bs)

            i += 1

            val_loss.update(loss, labels.size(0))
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

    miou = evaluator.Mean_Intersection_over_Union()
    accuracy = evaluator.Pixel_Accuracy()

    experiment.log_metric("val_epoch_loss", val_loss.avg, epoch=epoch, step=global_step)
    experiment.log_metric("val_epoch_accuracy", accuracy, epoch=epoch, step=global_step)
    experiment.log_metric("val_epoch_mIoU", miou, epoch=epoch, step=global_step)

    return global_step

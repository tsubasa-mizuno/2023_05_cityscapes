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

    for sample in val_loader:
        image, labels = sample["image"], sample["labels"]
        labels = labels.to(device)
        labels = labels.squeeze(dim=1)
        bs = len(image)
        if args.model == "Mask2Former":
            # target = [0] * 8
            for i in range(bs):
                image[i] = Image.open(image[i])
            inputs = processor(images=image, return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"].to(device)
            inputs["pixel_mask"] = inputs["pixel_mask"].to(device)
            # mask_labels_2 = torch.stack(
            #     [(labels == i).long() for i in range(args.num_class - 1)]
            # )
            # mask_labels_2 = mask_labels_2.long()

            # グランドトゥルースオブジェクトの数を取得
            labels = labels.long()
            num_target_boxes = labels.max() + 1

            class_labels = []
            mask_labels = []

            # 各オブジェクトに対して処理を行う
            for i in range(num_target_boxes):
                # オブジェクトの領域を特定
                mask = labels == i

                # マスクが空でないことを確認してからクラスラベルを取得
                if mask.any():
                    class_label = labels[mask].mode().values.item()
                else:
                    # Handle the case when the mask is empty
                    class_label = None

                # オブジェクトのマスクを保存
                mask_labels.append(mask)

                # オブジェクトのクラスラベルを保存
                class_labels.append(class_label)

            # Noneの値をフィルタリングする
            class_labels = [label for label in class_labels if label is not None]

            # 結果をテンソルに変換
            class_labels = (
                torch.tensor(class_labels).unsqueeze(1).expand(-1, bs).to(device)
            )
            mask_labels = torch.stack(mask_labels).float().to(device)

            outputs = model(
                **inputs, class_labels=class_labels, mask_labels=mask_labels
            )
            loss = outputs.loss
            # target_shape[256, 512]

        else:
            image = image.squeeze(dim=1)
            image = image.cuda().float()

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

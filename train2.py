"""training"""

from tqdm import tqdm
import torch
from PIL import Image
from util import AverageMeter


def train(
    model,
    processor,
    device,
    criterion,
    optimizer,
    loader,
    iters,
    epoch,
    experiment,
    evaluator,
    args,
    global_step,
):
    """_summary_

    Args:
        model (_type_): _description_
        processor (_type_): _description_
        device (_type_): _description_
        criterion (_type_): _description_
        optimizer (_type_): _description_
        loader (_type_): _description_
        iters (_type_): _description_
        epoch (_type_): _description_
        experiment (_type_): _description_
        evaluator (_type_): _description_
        args (_type_): _description_
        global_step (_type_): _description_

    Returns:
        _type_: _description_
    """

    evaluator.reset()

    model.train()

    train_loss = AverageMeter()

    with tqdm(loader, leave=False) as pbar_train:

        pbar_train.set_description("[train]")

        for sample in pbar_train:
            image, labels = (
                sample["image"],
                # B, C, H, W for U-Net, List of paths (string) for Mask2Former
                sample["labels"],  # B, 1, H, W
            )
            labels = labels.squeeze(dim=1)  # B, H, W

            if args.model == "Mask2Former":
                images = [Image.open(img) for img in image]
                inputs = processor(
                    images=images,
                    segmentation_maps=labels,
                    return_tensors="pt",
                )

                pixel_values = inputs["pixel_values"].to(device)  # B, 3, H, W,
                mask_labels = [labels_part.to(device) for labels_part in inputs["mask_labels"]]  # float32
                class_labels = [labels_part.to(device) for labels_part in inputs["class_labels"]]  # int64

                outputs = model(
                    pixel_values=pixel_values,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                )
                # 3ステップ目から出力画像からおかしいみたい？

                # デバッグコード（処理には要らない）
                target = []
                original_size = images[0].size[::-1]
                quarter_size = (original_size[0] // 4, original_size[1] // 4)
                target = processor.post_process_semantic_segmentation(
                    outputs, target_sizes=([quarter_size, quarter_size])
                )

                loss = outputs.loss

            else:
                labels = labels.to(device)
                image = image.to(device).float()
                target = model(image)  # B, C, H, W
                loss = criterion(target, labels)

            # Mask2Formerのlossが1枚目からすでにU-netより大きすぎ

            train_loss.update(loss, labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            experiment.log_metric("train_loss", loss, epoch=epoch, step=global_step)

    experiment.log_metric(
        "train_epoch_loss", train_loss.avg, epoch=epoch, step=global_step
    )
    global_step += 1

    return iters, global_step

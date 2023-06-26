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
    evaluator.reset()

    model.train()

    train_loss = AverageMeter()

    with tqdm(loader, leave=False) as pbar_train:
        pbar_train.set_description("[train]")

        for sample in pbar_train:
            image, labels = (
                sample["image"],  # B, C, H, W
                sample["labels"],  # B, 1, H, W
            )
            labels = labels.to(device)
            labels = labels.squeeze(dim=1)  # B, H, W
            bs = len(image)
            # labels_shape[8, 256, 512]

            optimizer.zero_grad()

            if args.model == "Mask2Former":
                images = [Image.open(img) for img in image]
                inputs = processor(images, return_tensors="pt")
                inputs["pixel_values"] = inputs["pixel_values"].to(device)
                inputs["pixel_mask"] = inputs["pixel_mask"].to(device)

                class_labels = []
                mask_labels = []

                for idx in range(bs):
                    class_label_of_b = labels[idx].unique().long()
                    class_labels.append(class_label_of_b)

                    mask_labels_of_b = []
                    for i in class_label_of_b:
                        mask = labels[idx] == i
                        mask_labels_of_b.append(mask.float())

                    mask_labels_of_b = torch.stack(mask_labels_of_b)
                    mask_labels.append(mask_labels_of_b)

                class_labels = [torch.tensor(labels) for labels in class_labels]

                outputs = model(
                    **inputs, class_labels=class_labels, mask_labels=mask_labels
                )
                outputs2 = model(**inputs)

                target = []
                original_size = images[0].size[::-1]
                quarter_size = (original_size[0] // 4, original_size[1] // 4)

                target = processor.post_process_semantic_segmentation(
                    outputs, target_sizes=([quarter_size, quarter_size])
                )
                predicted_semantic_map = processor.post_process_semantic_segmentation(
                    outputs2, target_sizes=([quarter_size, quarter_size])
                )[0]

                target = torch.stack(target)
                loss = outputs.loss
                # target_shape[256, 512]

            else:
                image = image.to(device).float()
                target = model(image)
                loss = criterion(target, labels.long())
                # target_shape[8, 3, 256, 512]

            # Mask2Formerのlossが1枚目からすでにU-netより大きすぎ

            train_loss.update(loss, labels.size(0))
            loss.backward()
            optimizer.step()
            experiment.log_metric("train_loss", loss, epoch=epoch, step=global_step)

            break

    experiment.log_metric(
        "train_epoch_loss", train_loss.avg, epoch=epoch, step=global_step
    )
    global_step += 1

    return iters, global_step

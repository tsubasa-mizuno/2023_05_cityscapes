"""make sample image from https://github.com/tamaki-lab/2022_05_Kamiya_Segmentation"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

classes_list = ["input", "GT_labels", "output", "background"]


def make_sample(image, labels, model, experiment, epoch, dataset):
    if dataset == "Cityscapes":
        c = []
        class_color = []
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    c.append([int(i * 255 / 5), int(j * 255 / 5), int(k * 255 / 5)])
        class_color = np.array(c)

    with torch.no_grad():
        y = model(image)

    pred = torch.argmax(y, dim=1)
    pred = pred.data.cpu().numpy()
    labels = labels.cpu().numpy()
    yy = F.softmax(y, dim=1)
    image = image.data.cpu().numpy()
    yy = yy.cpu().numpy()

    img_list = []

    image = image[0]
    v_img = (
        (image.transpose((1, 2, 0)) * [0.2023, 0.1994, 0.2010])
        + [0.4914, 0.4822, 0.4465]
    ) * 255
    v_img = np.uint8(v_img)
    img_list.append(v_img)

    result_img = np.transpose(pred, axes=[1, 2, 0])
    result_img = np.array(result_img).astype(np.uint8)
    labels = np.transpose(labels, axes=[1, 2, 0])
    labels = np.array(labels).astype(np.uint8)

    for i in range(0, class_color.shape[0]):
        result_img = result_img[:, :, :3]
        result_img[np.where((result_img == [i, i, i]).all(axis=2))] = class_color[i]
        labels = labels[:, :, :3]
        labels[np.where((labels == [i, i, i]).all(axis=2))] = class_color[i]

    # for i in range(0, class_color.shape[0]):
    #     result_img[np.where(np.all(np.asarray(result_img, dtype=np.uint8) == [i, i, i], axis=2))] = class_color[i]
    #     labels[np.where((labels == [i, i, i]).all(axis=2))] = class_color[i]

    labels = cv2.cvtColor(labels, cv2.COLOR_BGR2RGB)
    img_list.append(labels)
    if result_img.ndim == 2:
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    img_list.append(result_img)

    for i in range(yy.shape[1]):
        map = yy[:, i, :, :]
        map = np.transpose(map, axes=[1, 2, 0])
        map = np.uint8(map * 255)
        map = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
        map = cv2.applyColorMap(map, cv2.COLORMAP_JET)
        map = cv2.cvtColor(map, cv2.COLOR_BGR2RGB)
        img_list.append(map)

    row = 10
    col = 10
    plt.figure(figsize=(18, 10))
    num = 0
    while num < len(img_list):
        num += 1
        plt.subplot(row, col, num)
        plt.imshow(img_list[num - 1])

        if num - 1 < 4:
            plt.title("{}".format(classes_list[num - 1]))
        else:
            plt.title("item{}".format(num - 4))
        plt.axis("off")

    experiment.log_figure(figure=plt, figure_name="epoch_{}".format(epoch))


class_color = np.array(
    [
        [0, 0, 0],
        [85, 0, 0],
        [170, 0, 0],
        [255, 0, 0],
        [0, 85, 0],
        [85, 85, 0],
        [170, 85, 0],
        [255, 85, 0],
        [0, 170, 0],
        [85, 170, 0],
        [170, 170, 0],
        [255, 170, 0],
        [0, 255, 0],
        [85, 255, 0],
        [170, 255, 0],
        [255, 255, 0],
        [0, 0, 85],
        [85, 0, 85],
        [170, 0, 85],
        [255, 0, 85],
        [0, 85, 85],
        [85, 85, 85],
        [170, 85, 85],
        [255, 85, 85],
        [0, 170, 85],
        [85, 170, 85],
        [170, 170, 85],
        [255, 170, 85],
        [0, 255, 85],
        [85, 255, 85],
        [170, 255, 85],
        [255, 255, 85],
        [0, 0, 170],
        [85, 0, 170],
        [170, 0, 170],
        [255, 0, 170],
        [0, 85, 170],
        [85, 85, 170],
        [170, 85, 170],
        [255, 85, 170],
        [0, 170, 170],
    ]
)

class_color = class_color[:, ::-1]

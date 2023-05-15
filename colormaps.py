import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import os

# Cityscapesのカラーマップ
label_colors = np.array(
    [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    ],
    dtype=np.uint8,
)

# ラベル画像のファイルパス
labels_list = glob.glob(
    os.path.join(
        "/mnt/HDD4TB-3/mizuno/cityscapes/gtFine_trainvaltest/gtFine",
        "train/*/*_gtFine_labelIds.png",
    )
)
print(labels_list)

for label_file in labels_list:
    # ラベル画像を読み込む
    label = np.array(Image.open(label_file))
    print(label)

    if np.all(label <= 19):
        label_viz = label_colors[label]
        plt.imshow(label_viz)
        plt.show()

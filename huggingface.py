import torch
import torchvision
import torchvision.transforms.functional
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import numpy as np
import os
import torch.nn as nn


palette = (
    128,
    64,
    128,  # road
    244,
    35,
    232,  # sidewalk
    70,
    70,
    70,  # building
    102,
    102,
    156,  # wall
    190,
    153,
    153,  # fence
    153,
    153,
    153,  # pole
    250,
    170,
    30,  # traffic light
    220,
    220,
    0,  # traffic sign
    107,
    142,
    35,  # vegetation
    152,
    251,
    152,  # terrain
    70,
    130,
    180,  # sky
    220,
    20,
    60,  # person
    255,
    0,
    0,  # rider
    0,
    0,
    142,  # car
    0,
    0,
    70,  # truck
    0,
    60,
    100,  # bus
    0,
    80,
    100,  # train
    0,
    0,
    230,  # motorcycle
    119,
    11,
    32,  # bicycle
)

criterion = nn.CrossEntropyLoss(reduction="mean")

# load Mask2Former fine-tuned on Cityscapes semantic segmentation
processor = AutoImageProcessor.from_pretrained(
    "facebook/mask2former-swin-small-cityscapes-semantic"
)
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-small-cityscapes-semantic"
)

model = model.cuda()

# image = Image.open(
#     "/mnt/HDD10TB-1/mizuno/dataset/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/frankfurt/frankfurt_000001_016029_leftImg8bit.png"
# )

image = Image.open(
    "/mnt/HDD10TB-1/mizuno/dataset/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/munster/munster_000010_000019_leftImg8bit.png"
)

pil_labels = Image.open(
    "/mnt/HDD10TB-1/mizuno/dataset/cityscapes/gtFine_trainvaltest/gtFine/train/aachen/aachen_000030_000019_gtFine_labelIds.png"
)
inputs = processor(images=image, return_tensors="pt")

inputs["pixel_values"] = inputs["pixel_values"].cuda()
inputs["pixel_mask"] = inputs["pixel_mask"].cuda()

labels_numpy = np.array(pil_labels)
labels = torch.from_numpy(labels_numpy)
labels = labels.cuda()

with torch.no_grad():
    outputs = model(**inputs)

# model predicts class_queries_logits of shape `(batch_size, num_queries)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

# you can pass them to processor for postprocessing
target = processor.post_process_semantic_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)[0]
predicted_semantic_map = processor.post_process_semantic_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)[0]

target = target.float()

loss = criterion(target, labels.float())

# we refer to the demo notebooks for visualization (see "Resources" section in the Mask2Former docs)

# predicted_semantic_mapをnumpy配列に変換
# 2次元テンソルをNumPy配列に変換
tensor_array = predicted_semantic_map.cpu().numpy()

# データ型をuint8に変換
uint8_array = tensor_array.astype(numpy.uint8)

# PILイメージに変換
image = Image.fromarray(uint8_array)
# predict_img
image.putpalette(palette)
# predict_img_2

# PIL Imageに変換して保存
predicted_semantic_map_img = Image.fromarray(predicted_semantic_map)
predicted_semantic_map_img.save(
    "/mnt/HDD4TB-3/mizuno/2023_05_cityscapes/saveimg/predicted_semantic_map.png"
)

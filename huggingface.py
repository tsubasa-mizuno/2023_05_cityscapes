import requests
import torch
import torchvision
import torchvision.transforms.functional
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import numpy
import os


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

# load Mask2Former fine-tuned on Cityscapes semantic segmentation
processor = AutoImageProcessor.from_pretrained(
    "facebook/mask2former-swin-small-cityscapes-semantic"
)
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-small-cityscapes-semantic"
)

image = Image.open(
    "/mnt/HDD4TB-3/mizuno/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/aachen/aachen_000030_000019_leftImg8bit.png"
)
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# model predicts class_queries_logits of shape `(batch_size, num_queries)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

# pil_image = torchvision.transforms.functional.to_pil_image(masks_queries_logits)

# masks_queries_logits_numpy = numpy.argmax(
#     masks_queries_logits.cpu().detach().numpy(), axis=1
# )
# masks_queries_logits_numpy = masks_queries_logits_numpy[0].astype(
#     numpy.uint8
# )  # uint8に変換

# masks_img = Image.fromarray(masks_queries_logits_numpy)
# masks_img.putpalette(palette)


# you can pass them to processor for postprocessing
predicted_semantic_map = processor.post_process_semantic_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)[0]

# we refer to the demo notebooks for visualization (see "Resources" section in the Mask2Former docs)

# predicted_semantic_mapをnumpy配列に変換
# 2次元テンソルをNumPy配列に変換
tensor_array = predicted_semantic_map.numpy()

# データ型をuint8に変換
uint8_array = tensor_array.astype(numpy.uint8)

# PILイメージに変換
image = Image.fromarray(uint8_array)
image.putpalette(palette)

# PIL Imageに変換して保存
predicted_semantic_map_img = Image.fromarray(predicted_semantic_map)
predicted_semantic_map_img.save(
    "/mnt/HDD4TB-3/mizuno/2023_05_cityscapes/saveimg/predicted_semantic_map.png"
)


# predicted_semantic_map = predicted_semantic_map.masks_queries_logits

# predicted_semantic_map = numpy.argmax(
#     predicted_semantic_map.cpu().detach().numpy(), axis=1
# )
# predicted_semantic_map = predicted_semantic_map[0].astype(numpy.uint8)  # uint8に変換
# predicted_semantic_map = Image.fromarray(predicted_semantic_map)
# predicted_semantic_map.putpalette(palette)
# predicted_semantic_map.save(
#     os.path.join(
#         "/mnt/HDD4TB-3/mizuno/2023_05_cityscapes/saveimg/20230516_m2f_4",
#         f"{1}_{1}_TARGET_IMAGE.PNG",
#     )
# )

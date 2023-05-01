import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy

# # from args import get_args


# # class imagesave():
# #     def __init__(self, args):
# # 画像の読み込みと前処理
# transform = transforms.Compose(
#     [
#         transforms.Resize((256, 512)),  # 画像のリサイズ
#         transforms.ToTensor(),  # テンソルに変換
#     ]
# )

# # img_path = os.path.join(args.image_folder, "train/aachen/aachen_000000_000019_leftImg8bit.png")
# img_path = "/mnt/HDD4TB-3/mizuno/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png"
# img = Image.open(img_path).convert("RGB")
# img_tensor = transform(img)

# # テンソルを保存
# save_dir = "/mnt/HDD4TB-3/mizuno/202305_Cityscapes/saveimg"
# # if not os.path.exists(save_dir):
# #     os.makedirs(save_dir)
# torch.save(img_tensor, os.path.join(save_dir, "image.pt"))


def imagesave(target, labels, args, i):
    os.makedirs(args.save_dir, exist_ok=True)
    target = numpy.argmax(target.cpu().detach().numpy(), axis=1)  # one-hotを元のラベルに変換
    target_img = Image.fromarray(target[0].astype(numpy.uint8))
    target_img.putpalette(args.palette)
    target_img.save(os.path.join(args.save_dir, f"TARGET_IMAGE_{i}.PNG"))

    labels_img = Image.fromarray(labels[0].cpu().detach().numpy().astype(numpy.uint8))
    labels_img.putpalette(args.palette)
    labels_img.save(os.path.join(args.save_dir, f"LABELS_IMAGE_{i}.PNG"))

    # ペア画像がわかるように

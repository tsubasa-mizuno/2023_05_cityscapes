import os
from PIL import Image
import torch
import torchvision.transforms as transforms

# from args import get_args


# class imagesave():
#     def __init__(self, args):
# 画像の読み込みと前処理
transform = transforms.Compose(
    [
        transforms.Resize((256, 512)),  # 画像のリサイズ
        transforms.ToTensor(),  # テンソルに変換
    ]
)

# img_path = os.path.join(args.image_folder, "train/aachen/aachen_000000_000019_leftImg8bit.png")
img_path = "/mnt/HDD4TB-3/mizuno/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png"
img = Image.open(img_path).convert("RGB")
img_tensor = transform(img)

# テンソルを保存
save_dir = "/mnt/HDD4TB-3/mizuno/202305_Cityscapes/saveimg"
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
torch.save(img_tensor, os.path.join(save_dir, "image.pt"))

# 保存したテンソルを読み込んで画像を復元
loaded_tensor = torch.load(os.path.join(save_dir, "image.pt"))
loaded_img = transforms.functional.to_pil_image(loaded_tensor)
loaded_img.save(os.path.join(save_dir, "image_resized.png"))

# ペア画像がわかるように

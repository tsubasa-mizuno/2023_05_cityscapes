import os
from PIL import Image
import numpy


def imagesave(target, labels, args, i, count):
    os.makedirs(args.save_dir, exist_ok=True)
    palette = args.palette

    if args.model == "Unet":
        target_img = numpy.argmax(target.cpu().detach().numpy(), axis=1)
    else:
        target_img = numpy.argmax(target.cpu().detach().numpy(), axis=0)
    target_img = target_img.astype(numpy.uint8)  # uint8に変換
    # else:
    #     target = target.cpu().numpy()
    #     target_img = target.astype(numpy.uint8)  # uint8に変換
    target_img = Image.fromarray(target_img)
    target_img.putpalette(palette)
    target_img.save(os.path.join(args.save_dir, f"{i}_{count}_TARGET_IMAGE.PNG"))

    labels_img = Image.fromarray(labels[0].cpu().detach().numpy().astype(numpy.uint8))
    labels_img.putpalette(palette)
    labels_img.save(os.path.join(args.save_dir, f"{i}_LABELS_IMAGE.PNG"))

    # ペア画像がわかるように

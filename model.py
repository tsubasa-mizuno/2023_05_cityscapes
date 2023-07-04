"""specifing of models"""

import torch.nn as nn
import torch
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_factory(args):

    """defining using model"""

    if args.model == "Unet":
        model = torch.hub.load(
            "milesial/Pytorch-UNet", "unet_carvana", pretrained=args.pretrain, scale=0.5
        )
        in_channels = model.outc.conv.in_channels
        model.outc = nn.Conv2d(in_channels, args.output_channels, kernel_size=1)
        return model

    elif args.model == "Mask2Former":
        model_name = "facebook/mask2former-swin-small-cityscapes-semantic"
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        processor = AutoImageProcessor.from_pretrained(model_name)
        return model, processor

    else:
        raise ValueError("invalid args.model")

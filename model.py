import torch.nn as nn
import torch
import os


def model_factory(args):

    if args.model == 'Unet':
        model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=args.pretrain, scale=0.5)
        model.outc.conv = nn.Conv2d(args.input_channels, args.output_channels, kernel_size=1)

    else:
        raise ValueError("invalid args.model")

    return model

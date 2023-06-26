"""main function"""

import torch
import torch.nn as nn
from tqdm import tqdm
from args import get_args
from dataloader import dataset_facory
from model import model_factory
from util import Evaluator
from train2 import train
from val import val
from logger import logger_factory


def main():
    args = get_args()

    train_loader, val_loader = dataset_facory(args)

    processor = None

    if args.model == "Mask2Former":
        model, processor = model_factory(args)
    else:
        model = model_factory(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    experiment = logger_factory(args)
    evaluator = Evaluator(args.num_class)
    iters = 0
    global_step = 1

    with tqdm(range(args.num_epochs)) as pbar_epoch:
        for epoch in pbar_epoch:
            pbar_epoch.set_description("[Epoch {}]".format(epoch))

            iters, global_step = train(
                model,
                processor,
                device,
                criterion,
                optimizer,
                train_loader,
                iters,
                epoch,
                experiment,
                evaluator,
                args,
                global_step,
            )

            if epoch % args.val_epochs == 0:
                global_step = val(
                    model,
                    processor,
                    device,
                    criterion,
                    epoch,
                    val_loader,
                    evaluator,
                    experiment,
                    args,
                    global_step,
                )


if __name__ == "__main__":
    main()

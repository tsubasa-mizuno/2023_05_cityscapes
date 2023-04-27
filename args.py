import argparse


def get_args():
    parser = argparse.ArgumentParser(description='simple CNN model')
    # parser.add_argument('-d', '--dataset_name', type=str,
    #                     choices=['ARC', 'COCO'],
    #                     default='ARC',
    #                     help='name of dataset.')
    parser.add_argument(
        '--dataset',
        type=str,
        default='Cityscapes',
        choices=['Cityscapes', 'UCF', 'HMDB', 'Kinetics']
    )
    parser.add_argument('-m', '--model', type=str,
                        choices=['SegNet', 'Unet'],
                        default='Unet',
                        help='CNN model. default to SegNet')
    parser.add_argument('--pretrain', action='store_true',
                        help='use pretrained model')
    parser.add_argument('-b', '--batch_size', type=int, default=8,  # --train_num_batchsと同じ？
                        help='batch size. default to 8')
    parser.add_argument('-n', '--num_class', type=int, default=30,
                        help='number of class. default to 30')
    parser.add_argument('-e', '--num_epochs', type=int, default=10,
                        help='number of epochs. default to 10')
    parser.add_argument('-i', '--input_channels', type=int, default=64,
                        help='number of input_channels. default to 64')
    parser.add_argument('-o', '--output_channels', type=int, default=30,
                        help='number of output_channels. default to 30')
    parser.add_argument('-s', '--scale', type=float, default=0.5,
                        help='number of scale. default to 0.5')
    parser.add_argument('--val_epochs', type=int, default=10,
                        help='validation interval in epochs. default to 2')
    parser.add_argument('--save_epochs', type=int, default=20,
                        help='model save epochs. default to 10')

    parser.add_argument(
        '-l',
        '--labels',
        type=str,
        required=True,
        help='ground truth'
    )
    parser.add_argument(
        '-im',
        '--image',
        type=str,
        required=True,
        help='ground truth'
    )
    parser.add_argument(
        '-p',
        '--probability',
        type=float,
        default=1,
    )
    # parser.add_argument(
    #     '--train_num_batchs',
    #     type=int,
    #     default=16
    # )
    parser.add_argument(
        '--crop_size',
        type=int,
        default=256)

    # parser.add_argument(
    #     '--num_workers',
    #     type=int,
    #     default=1,
    #     help='number of dataloader workders. default 12'
    # )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU No. to be used for model. default 0'
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='how many GPUs to be used for model. default 1'
    )

    # dataset_path
    parser.add_argument(
        '--gtFine_folder',
        default='/mnt/HDD4TB-3/mizuno/cityscapes/gtFine_trainvaltest/gtFine',
        type=str
    )
    parser.add_argument(
        '--image_folder',
        default='/mnt/HDD4TB-3/mizuno/cityscapes/leftImg8bit_trainvaltest/leftImg8bit',
        type=str
    )
    parser.add_argument(
        '--dir_data_name',
        type=str,
        default="/mnt/HDD4TB-3/mizuno/2023_05_Cityscapes",
        help='model directory'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default="5",
        help='num_workers'
    )

    # parser.add_argument(
    #     '--model_dir',
    #     type=str,
    #     default="/mnt/HDD4TB-4/kamiya/SegNet-training/model",
    #     help='model directory'
    # )

    # ディレクトリ関連
    # parser.add_argument('ARC_path', type=str, default="./ARCdataset_png/",
    #                 help='ARC path directory')
    # parser.add_argument('ids_dir', type=str, default="/mnt/HDD4TB-4/kamiya/SegNet-training/ids",
    #                 help='ids directory')
    # parser.add_argument('coco_dir', type=str, default="/mnt/dataset/COCO",
    #                 help='coco directory')
    args = parser.parse_args()
    # print(args)

    return args

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="simple CNN model")
    # parser.add_argument('-d', '--dataset_name', type=str,
    #                     choices=['ARC', 'COCO'],
    #                     default='ARC',
    #                     help='name of dataset.')
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cityscapes",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="Unet",
        help="CNN model. default to Unet",
    )
    parser.add_argument("--pretrain", action="store_true", help="use pretrained model")
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=8,
        help="batch size. default to 8",
    )
    parser.add_argument(
        "-n", "--num_class", type=int, default=20, help="number of class. default to 20"
    )
    parser.add_argument(
        "-e",
        "--num_epochs",
        type=int,
        default=40,
        help="number of epochs. default to 40",
    )
    parser.add_argument(
        "-i",
        "--input_channels",
        type=int,
        default=3,
        help="number of input_channels. default to 3",
    )
    parser.add_argument(
        "-o",
        "--output_channels",
        type=int,
        default=20,
        help="number of output_channels. default to 20",
    )

    # 学習率
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        default=0.01,
        help="number of scale. default to 0.01",
    )

    parser.add_argument(
        "--val_epochs",
        type=int,
        default=5,
        help="validation interval in epochs. default to 5",
    )
    parser.add_argument(
        "--save_epochs", type=int, default=10, help="model save epochs. default to 10"
    )

    parser.add_argument("-l", "--labels", type=str, required=True, help="ground truth")
    parser.add_argument("-im", "--image", type=str, required=True, help="ground truth")
    parser.add_argument(
        "-p",
        "--probability",
        type=float,
        default=1,
    )
    # parser.add_argument(
    #     '--train_num_batchs',
    #     type=int,
    #     default=16
    # )
    parser.add_argument("--crop_size", type=int, default=256)

    # parser.add_argument(
    #     '--num_workers',
    #     type=int,
    #     default=1,
    #     help='number of dataloader workders. default 12'
    # )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU No. to be used for model. default 0"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="how many GPUs to be used for model. default 1",
    )

    # dataset_path
    parser.add_argument(
        "--gtFine_dir",
        default="/mnt/HDD4TB-3/mizuno/cityscapes/gtFine_trainvaltest/gtFine",
        type=str,
    )
    parser.add_argument(
        "--image_dir",
        default="/mnt/HDD4TB-3/mizuno/cityscapes/leftImg8bit_trainvaltest/leftImg8bit",
        type=str,
    )
    parser.add_argument(
        "--save_dir",
        default="/mnt/HDD4TB-3/mizuno/2023_05_cityscapes/saveimg",
        type=str,
    )

    parser.add_argument(
        "--dir_data_name",
        type=str,
        default="/mnt/HDD4TB-3/mizuno/2023_05_Cityscapes",
        help="model directory",
    )

    parser.add_argument("--workers", type=int, default="8", help="num_workers")

    parser.add_argument(
        "--palette",
        default=[
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
        ],
    )

    parser.add_argument(
        "--label_dict",
        type=dict,
        default={
            0: 19,  # unlabeled
            1: 19,  # ego vehicle
            2: 19,  # rectification border
            3: 19,  # out of roi
            4: 19,  # static
            5: 19,  # dynamic
            6: 19,  # ground
            7: 0,  # road
            8: 1,  # sidewalk
            9: 19,  # parking
            10: 19,  # rail track
            11: 2,  # building
            12: 3,  # wall
            13: 4,  # fence
            14: 19,  # guard rail
            15: 19,  # bridge
            16: 5,  # tunnel
            17: 19,  # pole
            18: 6,  # traffic light
            19: 7,  # traffic sign
            20: 8,  # vegetation
            21: 9,  # terrain
            22: 10,  # sky
            23: 11,  # person
            24: 12,  # rider
            25: 19,  # car
            26: 13,  # truck
            27: 14,  # bus
            28: 19,  # caravan
            29: 19,  # trailer
            30: 15,  # train
            31: 16,  # motorcycle
            32: 17,  # bicycle
            -1: 19,  # ignore
        },
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

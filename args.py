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
        choices=["Unet", "Mask2Former"],
        default="Unet",
        help="CNN model.",
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
        default=30,
        help="number of epochs. default to 30",
    )
    parser.add_argument(
        "-i",
        "--input_channels",
        type=int,
        default=35,
        help="number of input_channels. default to 35",
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
        default=0.005,
        help="number of scale. default to 0.005",
    )

    parser.add_argument(
        "--val_epochs",
        type=int,
        default=5,
        help="validation interval in epochs. default to 5",
    )
    parser.add_argument(
        "--save_epochs", type=int, default=5, help="model save epochs. default to 5"
    )

    parser.add_argument("-l", "--labels", type=str, required=True, help="ground truth")
    parser.add_argument("-im", "--image", type=str, required=True, help="ground truth")
    parser.add_argument(
        "-p",
        "--probability",
        type=float,
        default=1,
    )

    parser.add_argument("--crop_size", type=int, default=384)

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU No. to be used for model. default 0",
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
        default="/mnt/HDD4TB-3/mizuno/2023_05_cityscapes/saveimg/20230514_64_1",
        type=str,
    )

    parser.add_argument(
        "--dir_data_name",
        type=str,
        default="/mnt/HDD4TB-3/mizuno/2023_05_Cityscapes",
        help="model directory",
    )

    parser.add_argument("--workers", type=int, default="16", help="num_workers")

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
            16: 19,  # tunnel
            17: 5,  # pole
            18: 19,  # polegroup
            19: 6,  # traffic light
            20: 7,  # traffic sign
            21: 8,  # vegetation
            22: 9,  # terrain
            23: 10,  # sky
            24: 11,  # person
            25: 12,  # rider
            26: 13,  # car
            27: 14,  # truck
            28: 15,  # bus
            29: 19,  # caravan
            30: 19,  # trailer
            31: 16,  # train
            32: 17,  # motorcycle
            33: 18,  # bicycle
            -1: -1,  # ignore
        },
    )

    # disabling comet for debugging
    parser.add_argument(
        "--disable_comet",
        "--no_comet",
        dest="disable_comet",
        action="store_true",
        help="do not use comet.ml (default: use comet)",
    )
    parser.set_defaults(disable_comet=False)

    args = parser.parse_args()

    return args

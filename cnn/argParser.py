# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------

import argparse

def build_parser():
    parser = argparse.ArgumentParser(
        description="Arguments for CNN experiments."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=100,
        help="Mini-batch size for training and validation.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="SGD learning rate.",
    )
    parser.add_argument(
        "--hidden-size",
        dest="hidden_size",
        type=int,
        default=22,
        help="Number of nodes in the dense hidden layer.",
    )
    parser.add_argument(
        "--num-samples",
        dest="num_samples",
        type=int,
        default=1000,
        help="Number of samples per class (circles and rectangles each).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=73,
        help="Random seed.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="Debug verbosity level.",
    )

    return parser


def parse_args():
    return build_parser().parse_args()


args = parse_args()

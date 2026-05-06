# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------


import argparse

# This is the only file written using an LLM help.
# rest of the code in this work are hand coded with no
# AI assistance with the objective to make the learning stick!.

def build_parser():
    parser = argparse.ArgumentParser(
        description="Arguments for RNN language model experiments."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input npz music file used for training.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=32,
        help="Number of training samples per batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=33,
        help="Random seed for initialization.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="Debug verbosity level (1=output, 2=error, 3=warn, 4=debug, 5=info).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate for optimizer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--seq_len",
        dest="seq_len",
        type=int,
        default=128,
        help="Number of 1/8th time-steps in each sample",
    )
    parser.add_argument(
        "--num_samples",
        dest="num_samples",
        type=int,
        default=2000,
        help="Number of music samples to generate.",
    )
    parser.add_argument(
        "--hidden-dim",
        dest="hidden_dim",
        type=int,
        default=256,
        help="hidden state dimension",
    )
    parser.add_argument(
        "--model-file",
        dest="model_file",
        type=str,
        default=None,
        help="Path to saved model file. If provided, skips training and runs inference only.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam"],
        help="Optimizer to use: sgd or adam.",
    )
    parser.add_argument(
        "--num-layers",
        dest="num_layers",
        type=int,
        default=1,
        help="Number of stacked GRU layers.",
    )

    return parser


def parse_args():
    return build_parser().parse_args()


args = parse_args()

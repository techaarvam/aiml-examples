
import argparse

# This is the only file written using an LLM help.
# rest of the code in this work are hand coded with no 
# AI assistance with the objective to make the learning stick!.

def build_parser():
    parser = argparse.ArgumentParser(
        description="Arguments for ANN experiments."
    )

    parser.add_argument(
        "--hidden-size",
        dest="hidden_size",
        type=int,
        default=300,
        help="Number of nodes in the hidden layer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr1",
        type=float,
        default=0.9,
        help="Learning rate for layer 1.",
    )
    parser.add_argument(
        "--lr2",
        type=float,
        default=0.003,
        help="Learning rate for layer 2.",
    )
    parser.add_argument(
        "--weight-scale",
        dest="weight_scale",
        type=float,
        default=0.1,
        help="Scale factor for random weight initialization.",
    )
    parser.add_argument(
        "--bias-scale",
        dest="bias_scale",
        type=float,
        default=0.0,
        help="Scale factor for bias initialization.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Enable input normalization.",
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
        default=5,
        help="Debug verbosity level.",
    )
    parser.add_argument(
        "--report-every",
        dest="report_every",
        type=int,
        default=1,
        help="Print metrics every N epochs.",
    )

    return parser


def parse_args():
    return build_parser().parse_args()


args = parse_args()

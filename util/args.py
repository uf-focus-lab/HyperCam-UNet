# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# TODO: Add description
# ---------------------------------------------------------
import argparse
from os.path import isdir
from random import randint
from .env import Path, RUN_PATH

parser = argparse.ArgumentParser(
    prog="CUDA_DEV[ICE]=* main.py [run-all | train | test]",
    description="Train and test Torch models",
    epilog="Author: Yuxuan Zhang (zhangyuxuan@ufl.edu)",
)

parser.add_argument("-m", "--model", type=str, required=True, help="Name of the model")

parser.add_argument(
    "-M",
    "-f",
    "--train-mode",
    type=str,
    default=["generic"],
    action="append",
    nargs="*",
    help="Training mode to be passed towards model.",
)

parser.add_argument(
    "-p",
    "--preprocess",
    type=str,
    default=[],
    nargs="*",
    help="Preprocessing flags.",
)

parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")

parser.add_argument("-b", "--batch-size", type=int, default=10, help="Batch size")

# parser.add_argument(
#     '-l', '--lossFunction',
#     type=str, default="",
#     help="Loss function (not implemented yet)"
# )

parser.add_argument(
    "-r", "--learning-rate", type=float, default=1e-2, help="Learning rate"
)

parser.add_argument("-d", "--weight-decay", type=float, default=0, help="Weight Decay")

# parser.add_argument(
#     '-k', '--kFoldRatio',
#     type=float, default=0.8,
#     help="Split ratio of k-Fold cross validation"
# )

parser.add_argument(
    "-L",
    "--load",
    type=str,
    default=[],
    action="append",
    nargs="*",
    help="Path to load pre-trained model",
)

parser.add_argument(
    "-s",
    "--seed",
    type=int,
    default=randint(0, 0xFFFFFFFF),
    help="Manually specify random seed, randomly generated if not specified",
)

parser.add_argument("command", nargs="*", type=str)

parser.add_help = True


def flatten(l: list[list]) -> list:
    result = []
    for item in l:
        if isinstance(item, list):
            result += flatten(item)
        else:
            result.append(item)
    return result


def load_list(args: list[list[str]]):
    if len(args) <= 0:
        return None
    at_dict = {}
    for s in flatten(args):
        s = s.split(":")
        remap = None
        if len(s) == 1:
            """[RUN_ID]"""
            key = ""
            (val,) = s
        elif len(s) == 2:
            """[MODEL_NAME]:[RUN_ID]"""
            key, val = s
        else:
            """[MODEL_NAME]:[RUN_ID]:[REMAP]"""
            assert len(s) == 3, ":".join(s)
            key, val, remap = s
        at_dict[key] = (val, remap)
    return at_dict


def normalizeRunPaths(kwargs) -> dict[str, tuple[Path, str | None]]:
    if not isinstance(kwargs, dict):
        return kwargs
    for key in kwargs:
        runID, remap = kwargs[key]
        path = RUN_PATH / runID
        assert isdir(path), path
        kwargs[key] = path, remap
    return kwargs


def normalizeFlag(s: str) -> str:
    return s.strip().upper().replace("-", "_")


if __name__ == "__main__":
    parser.print_help()
    print(parser.parse_args(["-m", "model-name"]))
else:
    ARGS = parser.parse_args()
    command = ARGS.command
    CMD = command[0] if len(command) else "run-all"
    RUN_TRAIN = CMD in ["run-all", "train"]
    RUN_TEST = CMD in ["run-all", "test"]
    model: str = ARGS.model
    train_mode = list(map(normalizeFlag, flatten(ARGS.train_mode)))
    preprocess = list(map(normalizeFlag, flatten(ARGS.preprocess)))
    epochs: int = ARGS.epochs
    batch_size: int = ARGS.batch_size
    learning_rate: float = ARGS.learning_rate
    weight_decay: float = ARGS.weight_decay
    load: dict[str, str] | None = normalizeRunPaths(load_list(ARGS.load))
    seed: int = ARGS.seed
    model = ARGS.model

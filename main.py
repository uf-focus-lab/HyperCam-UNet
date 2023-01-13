#!python3
# Python packages
import sys
import argparse
from os.path import exists
# PIP Packages
import torch
import numpy as np
# User includes
from dataset import DataSet
from module import Module
import env
import util
# Model imports
import models.U_Net as U_Net

# Model map
MODELS = {
    "U_Net": U_Net
}

# Arguments
parser = argparse.ArgumentParser(
    prog='main.py [run-all | train | test]',
    description='Train and test ML models with given parameters',
    epilog='Author: Yuxuan Zhang (zhangyuxuan@ufl.edu)')
parser.add_argument('-m', '--model', type=str, default="U_Net", help="Name of the model ({})".format(", ".join(MODELS.keys())))
parser.add_argument('-e', '--epochs', type=int, default=10, help="Number of epoches")
parser.add_argument('-b', '--batchSize', type=int, default=10, help="Batch size")
parser.add_argument('-l', '--lossFunction', type=str, default="", help="Loss function (not implemented yet)")
parser.add_argument('-r', '--learningRate', type=float, default=1e-6, help="Learning rate")
parser.add_argument('-k', '--kFoldRatio', type=float, default=0.8, help="Split ratio of k-Fold corss validation")
parser.add_argument('-L', '--load', type=str, default=None, help="Path to load pre-trained model")
parser.add_argument('-S', '--shuffle', type=float, default=-1, help="Flag to re-shuffle train/test lists")
parser.add_help = True

# Extract commands
flag_run_train = False
flag_run_test = False
try:
    CMD = sys.argv[1] if len(sys.argv) > 1 else None
    cmd = CMD.lower() if type(CMD) == str else None
    argList = sys.argv[2:]
    if (cmd == "run-all" or CMD is None or CMD.startswith('-')):
        flag_run_train = True
        flag_run_test = True
        if (CMD is not None and CMD.startswith('-')):
            argList.insert(0, CMD)
    elif (cmd == "train"):
        flag_run_train = True
    elif (cmd == "test"):
        flag_run_test = True
    else:
        print("Error: invalid command \"{}\"".format(CMD), file=sys.stderr)
        raise Exception
    # Parse additional args
    args = parser.parse_args(argList)
except Exception as e:
    parser.print_help()
    sys.exit(1)

# Constants
RUN_ID = util.uniq(env.RUN_PATH)
WORK_DIR = env.RUN_PATH / RUN_ID
print("Run ID:", RUN_ID)

# Initialize run log
RUN_LOG_PATH = env.VAR_PATH / "run.log"
if not exists(RUN_LOG_PATH):
    RUN_LOG_PATH.touch()
with open(RUN_LOG_PATH, "a") as f:
    print("{} | {}".format(RUN_ID, " ".join(sys.argv)), file=f)

# Initialize model and loss function
Model = MODELS[args.model].Model
lossFunction = MODELS[args.model].lossFunction

# Initialize Paths
env.ensureDir(WORK_DIR)
with open(WORK_DIR / "log.txt", 'w') as f:
    print("{} | {}".format(RUN_ID, " ".join(sys.argv[3:])), file=f)

# Initialize parameters
kFoldRatio = args.kFoldRatio
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batchSize
LOSS_FN = args.lossFunction
kFoldRatio = args.kFoldRatio
LR = args.learningRate

# Load datasets
S = args.shuffle
flag_reload, shuffle_ratio = (True, S) if S > 0 and S < 1.0 else (False, 0.1)
train_set, test_set = DataSet.load(flag_reload, shuffle_ratio)

# Create Model
model: Module = Model(env.DEVICE, train_set.sample())
model.to(env.DEVICE)

# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

# Record all free parameters
with open(WORK_DIR / "parameters.txt", "w") as f:
    print("Device        =", env.DEVICE, file=f)
    print("Num Epochs    =", NUM_EPOCHS, file=f)
    print("Batch Size    =", BATCH_SIZE, file=f)
    print("Loss Fn       =", str(LOSS_FN), file=f)
    print("Learning Rate =", LR, file=f)
    print(file=f)


if args.load is not None:
    load_path = env.RUN_PATH / args.load
    print(">> Loading model from ", load_path)
    model.load_state_dict(torch.load(str(load_path / "model.pkl")))
    model.eval()
    print(">> Loading optimizer from ", load_path)
    optimizer.load_state_dict(torch.load(str(load_path / "optim.pkl")))

# Run train on demand
if flag_run_train:
    print("========== Model Definition ==========")
    print(model)
    print(model, file=open(WORK_DIR / "Model.txt", "w"))
    print("========== Training Model ==========")
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE)
    model.run_train(train_loader, optimizer, lossFunction, NUM_EPOCHS, kFoldRatio, file=open(WORK_DIR / "train.log.txt", 'w'), work_path=WORK_DIR)
    # Save model to run dir
    torch.save(model.state_dict(), str(WORK_DIR / "model.pkl"))
    torch.save(optimizer.state_dict(), str(WORK_DIR / "optim.pkl"))
    print("========== Running Test on TRAIN SET ==========")
    model.run_test(train_set, file=open(WORK_DIR / "test.log.txt", 'w'), work_path=WORK_DIR / "train_results")

if flag_run_test:
    print("========== Running Test on TEST SET ==========")
    model.run_test(test_set, file=open(WORK_DIR / "test.log.txt", 'w'), work_path=WORK_DIR / "test_results")
# Mark as successful run
(WORK_DIR / "000_SUCCESS").touch()

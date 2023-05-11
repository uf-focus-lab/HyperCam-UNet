#!python
# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# Main entry for running train and test on specified model.
# Run this script with no parameters to see help message.
# ---------------------------------------------------------
# Python Modules
import random

# PIP Modules
import torch
from torch.utils.data import DataLoader

# User Modules
from util.dataset import DataSet
from lib import Run, Module
from util.dataset import DataSet
from util.device import DEVICE
from util.optimizer import optimizer
import util.args as args

# Model imports
from models import MODELS

# Initialize datasets
train_set, test_set = DataSet.load()
# Create context
with Run() as run:
    # Initialize random seed (IMPORTANT!!!)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Initialize model
    run.log(banner="Model Initialization")
    Model = MODELS[args.model]
    model: Module = Model(run, DEVICE, train_set.sample(device=DEVICE))
    model.to(DEVICE)
    # Model too large to be displayed
    run.log(model, file="model.txt", visible=False)
    # Record all free parameters
    run.log(banner="Free Parameters")
    run.log("Device        =", DEVICE)
    run.log("Training Mode =", args.train_mode)
    run.log("Num Epochs    =", args.epochs)
    run.log("Batch Size    =", args.batch_size)
    run.log("Random Seed   =", args.seed)
    # Check for previous model to load
    if args.load is not None:
        run.log(banner="Loading Model States")
        model.load(run, **args.load)
    # ================================= TRAIN =================================
    if args.RUN_TRAIN:
        # Initialize optimizer
        optim = optimizer(torch.optim.Adam)(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        run.log(optim, file="optim.txt", visible=False)
        if args.load:
            run.log(banner="Loading Optimizer States")
            optim.load(run, **args.load)
        with run.context(
            "train", optimizer=optim, epochs=args.epochs, train_mode=args.train_mode
        ) as ctx:
            train_loader = DataLoader(train_set, batch_size=args.batch_size)
            ctx.log(banner="Training Model")
            model.run(ctx, train_loader)
        # Save model prediction on training set
        run.log(banner="Saving States")
        model.save(run)
        optim.save(run)
        # Run test on training set
        with run.context("train") as ctx:
            ctx.log(banner="Running Prediction on TRAINING SET")
            model.run(ctx, train_set)
    # ================================== TEST =================================
    if args.RUN_TEST:
        with run.context("test") as ctx:
            ctx.log(banner="Running Prediction on TEST SET")
            model.run(ctx, test_set)
    # Visualize
    # if "visualize" in args.command:
    #     with torch.no_grad():
    #         tools.vis_pred.ui(run, model, train_set, test_set)
    # Congratulations!
    run.log(f"RUN<{run.id}> completed!", banner="Success")

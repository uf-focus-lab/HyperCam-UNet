# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# TODO: Add description
# ---------------------------------------------------------
# PIP Modules
import sys
from glob import glob
from math import floor
from typing import Tuple
from random import shuffle, random
from pathlib import Path
from os.path import basename, exists, isfile
from functools import cache

# PIP modules
import torch
from torch.utils.data import Dataset as TorchDataset
import numpy as np

# Custom modules
import cvtb
from util.env import VAR_PATH, DATA_PATH
import util.preprocess as preprocess

RAW_DATA_PATH = DATA_PATH / "raw"
REF_DATA_PATH = DATA_PATH / "ref"

TRAIN_SET_LIST_PATH = VAR_PATH / "train.list"
TEST_SET_LIST_PATH = VAR_PATH / "test.list"

Sample_t = Tuple[torch.Tensor, torch.Tensor]


def getList(reload: bool = False, ratio: float = 0.1):
    # Generate new file list
    if reload or not exists(TRAIN_SET_LIST_PATH) or not exists(TEST_SET_LIST_PATH):
        return listGen(ratio)

    # Load pre-defined file list
    def load(path):
        f = open(path, "r").readlines()
        return [_.replace("\n", "") for _ in f]

    return load(TRAIN_SET_LIST_PATH), load(TEST_SET_LIST_PATH)


def listGen(ratio: float = 0.1):
    fList = map(basename, glob(str(RAW_DATA_PATH / "*.npy")))
    nameList = [_.replace(".npy", "") for _ in fList]
    # Check for existance of corresponding reference files
    for name in nameList:
        assert exists(REF_DATA_PATH / f"{name}.npy"), name
    # Randomly split the dataset into training set and test set
    shuffle(nameList)
    NUM_TEST = round(len(nameList) * ratio)
    TRAIN_LIST = nameList[NUM_TEST:]
    TEST_LIST = nameList[:NUM_TEST]
    with open(TRAIN_SET_LIST_PATH, "w") as f:
        f.write("\n".join(nameList[NUM_TEST:]))
    with open(TEST_SET_LIST_PATH, "w") as f:
        f.write("\n".join(nameList[:NUM_TEST]))
    return TRAIN_LIST, TEST_LIST


@cache
def load(fileID: str):
    fileName = f"{fileID}.npy"

    def _load(path: Path):
        assert isfile(path), path
        # Load data matrix
        data = cvtb.types.F32(np.load(str(path)))
        # Transform to torch tensor
        h, w, d = data.shape
        data = torch.from_numpy(data.astype(np.float32))
        # Rearrange axes (h, w, d) => (d, w, h)
        data = data.transpose(0, 2).view((d, w, h))
        return data

    data = _load(RAW_DATA_PATH / fileName)
    truth = _load(REF_DATA_PATH / fileName)
    return data, truth


class DataSet(TorchDataset):
    """Custom dataset implementation"""

    @classmethod
    def load(cls, *flags: str, reload: bool = False, ratio: float = 0.1, device=None):
        return [cls(l, *flags, device=device) for l in getList(reload, ratio)]

    def sample(self, count: int = 1):
        # Randomly return [count] items as the sample
        idx = list(range(len(self)))
        shuffle(idx)
        idx = idx[:count]
        # Load data
        sample = [self.__getitem__(i) for i in idx]
        data, truth, fileID = zip(*sample)
        data = torch.stack(data)
        truth = torch.stack(truth)
        return data, truth

    def __init__(self, idList, *flags: str, device=None):
        """
        load dataset from a given data set collection
        """
        if len(idList) == 0:
            print("No data given", file=sys.stderr)
            raise Exception
        self.data_points = idList
        self.flags = flags
        self.device = device

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        fileID = self.data_points[idx]
        data, truth = load(fileID)
        # Send to device
        if self.device is not None:
            data = data.to(self.device)
            truth = truth.to(self.device)
        # Apply preprocessing
        if "REMOVE_SPOTS" in self.flags:
            data = preprocess.remove_spots(data)
        return data, truth, fileID

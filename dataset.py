import sys
from glob import glob
from os.path import basename, exists
from random import shuffle, random
from math import floor
from typing import Tuple
# Custom imports
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from env import RAW_DATA_PATH, REF_DATA_PATH, VAR_PATH

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
    with open(TRAIN_SET_LIST_PATH, 'w') as f:
        f.writelines(nameList[NUM_TEST:])
    with open(TEST_SET_LIST_PATH, 'w') as f:
        f.writelines(nameList[:NUM_TEST])
    return TRAIN_LIST, TEST_LIST


class DataSet(TorchDataset):
    """Custom dataset implementation"""


    @classmethod
    def load(cls, reload: bool = False, ratio: float = 0.1):
        train_list, test_list = getList(reload, ratio)
        return cls(train_list), cls(test_list)


    def sample(self):
        # Randomly return an item as the sample
        idx = floor(random() * self.__len__())
        data, truth, _ = self.__getitem__(idx)
        d, w, h = data.shape
        data = data.view((-1, d, w, h))
        d, w, h = truth.shape
        truth = truth.view((-1, d, w, h))
        return data, truth


    def __init__(self, idList):
        """
        load dataset from a given data set collection
        """
        if len(idList) == 0:
            print("No data given", file=sys.stderr)
            raise Exception
        self.data_points = idList


    def __len__(self):
        return len(self.data_points)


    def __getitem__(self, idx):
        fileID = self.data_points[idx]
        fileName = fileID + ".npy"
        # Load numpy array and transform into torch tensor
        def load(path):
            # Load data matrix
            data = np.load(str(path)).astype(np.float32)
            data = data / np.max(data)
            h, w, d = data.shape
            data = torch.from_numpy(data.astype(np.float32))
            # Rearrange axes (h, w, d) => (d, w, h)
            data = data.transpose(0, 2).view((d, w, h))
            return data
        # Load corresponding label vector
        data = load(str(RAW_DATA_PATH / fileName))
        truth = load(str(REF_DATA_PATH / fileName))
        return data, truth, fileID

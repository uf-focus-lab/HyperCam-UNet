#!python
# Python Packages
import sys
from os.path import exists, dirname, basename, abspath
from tkinter.filedialog import askopenfilename
# PIP Packages
import numpy as np
from PIL import Image
# Custom Packages
import env

def isNpMtx(f): return type(f) == str and f.endswith(".npy") and exists(f)


def askForMtx(): return askopenfilename(
	defaultextension=".npy", initialdir=str(env.RUN_PATH))


RESULT_PATH = sys.argv[-1] if isNpMtx(sys.argv[-1]) else askForMtx()
if RESULT_PATH is None:
	sys.exit(0)
if not exists(RESULT_PATH):
	print("File \"{}\" does not exist".format(RESULT_PATH), file=sys.stderr)
	sys.exit(1)

CAPTURE_PREFIX = abspath(dirname(RESULT_PATH))	\
	.replace(abspath(env.RUN_PATH) + '/', '')	\
	.replace('_results', '')					\
	.replace('/', '_')

CAPTURE = env.ensureDir(env.VAR_PATH / "capture")
CAPTURE_DIR = env.ensureDir(CAPTURE / CAPTURE_PREFIX)

ID = basename(RESULT_PATH).replace(".npy", "")

def removeSpots(shape, img: np.ndarray, levels: list = []):
    h, w, d = shape
    n = len(levels)
    layers = [None for _ in levels]
    bri = np.average(img, axis=(0, 1))
    min_img = np.min(img.astype(np.float64) / bri, axis=2) / 255
    for i, L in zip(range(n), levels):
        layer = img[:, :, i].astype(np.float64) / 255
        mask = layer > L
        layer = layer * mask + min_img * bri[i] * (1 - mask)
        layer = np.round(255 * layer / L).astype(np.uint8)
        layer = np.array(Image.fromarray(layer).resize((w, h)))
        layers[i] = layer
    return np.stack(layers, axis=2)


def normalize(img: np.ndarray)->np.ndarray[np.uint8]:
    if img.dtype != np.uint8:
        img = img.astype(np.float64)
        return (img * 255 / np.max(img)).astype(np.uint8)
    return img

PRD = normalize(np.load(RESULT_PATH))

RAW = normalize(removeSpots(
    PRD.shape,
    np.load(str(env.DATA_PATH / (ID + ".npy"))),
    levels=[0.25, 0.25, 0.25, 0.25, 0.2, 0.25, 0.25, 0.9]
))

REF = normalize(np.load(str(env.DATA_PATH / (ID + "_REF.npy"))))

#!python3
# Python Packages
import sys
from os.path import exists, basename
# PIP Packages
import cv2 as cv
import numpy as np
# Custom Packages
import env

if len(sys.argv) <= 1:
	print("usage: check.py path/to/result.npy", file=sys.stderr)
	sys.exit(1)

RESULT_PATH = sys.argv[1]

if not exists(RESULT_PATH):
	print("File \"{}\" does not exist".format(RESULT_PATH), file=sys.stderr)
	sys.exit(1)

ID = basename(RESULT_PATH).replace(".npy", "")
result = np.load(RESULT_PATH)
reference = np.load(str(env.DATA_PATH / (ID + "_REF.npy")))
reference = (reference * 255 / np.max(reference)).astype(np.uint8)

h, w, d = result.shape

WINDOW_NAME = "Inspecting {}".format(ID)
cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)
cv.startWindowThread()
cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_TOPMOST, 1)
for i in range(d):
	img1 = reference[:, :, i]
	img2 = result[:, :, i]
	# Compute diff image
	R = (img1 - img2) * (img1 > img2)
	B = (img2 - img1) * (img2 > img1)
	G = np.zeros(R.shape, dtype=np.uint8)
	diff = np.stack((B, G, R), axis=2)
	# Convert imgs to gray RGB
	img1, img2 = [np.stack((_, _, _), axis=2) for _ in [img1, img2]]
	blank = np.zeros((h, 20, 3), dtype=np.uint8)
	print("Viewing band #{} ({})".format(i, np.max(img2)))
	cv.imshow(WINDOW_NAME, np.concatenate((img1, blank, img2, blank, diff), axis=1))
	key = cv.waitKey(100)
	if key == 'q' or key == 27:
		break
cv.destroyAllWindows()
cv.waitKey(10)
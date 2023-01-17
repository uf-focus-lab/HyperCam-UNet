#!python3
# Python Packages
import sys
import argparse
import subprocess
from os.path import exists, basename
from tkinter.filedialog import askopenfilename
# PIP Packages
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# Custom Packages
import env
PROGRNAM_NAME = sys.argv[0]
RESULT_PATH = sys.argv[1] if len(sys.argv) > 1 else askopenfilename(defaultextension=".npy", initialdir=str(env.RUN_PATH))
# Inspection
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--plot', type=str, default=None)
args = parser.parse_args(sys.argv[2:])

if not exists(RESULT_PATH):
	print("File \"{}\" does not exist".format(RESULT_PATH), file=sys.stderr)
	sys.exit(1)

ID = basename(RESULT_PATH).replace(".npy", "")
result = np.load(RESULT_PATH)
reference = np.load(str(env.DATA_PATH / (ID + "_REF.npy")))
reference = (reference * 255 / np.max(reference)).astype(np.uint8)

h, w, d = result.shape

if args.plot is not None:
	x, y = [int(_) for _ in args.plot.split(',')]
	line1 = np.reshape(reference[x, y, :], (-1))
	line2 = np.reshape(result[x, y, :], (-1))
	plt.plot(line1, 'r-')
	plt.plot(line2, 'b-')
	plt.show()
	sys.exit(0)


WINDOW_NAME = "Inspecting {}".format(ID)
cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)
cv.startWindowThread()
cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_TOPMOST, 1)

MARGIN = 20
user_selection = None
i = 0


def view(_i=None):
	global i
	if _i is not None:
		i = _i
	img1 = reference[:, :, i]
	img2 = result[:, :, i]
	# Compute diff image
	R = (img1 - img2) * (img1 > img2)
	B = (img2 - img1) * (img2 > img1)
	G = np.zeros(R.shape, dtype=np.uint8)
	diff = np.stack((B, G, R), axis=2)
	# Convert imgs to gray RGB
	img1, img2 = [np.stack((_, _, _), axis=2) for _ in [img1, img2]]
	if user_selection is not None:
		img1 = cv.drawMarker(img1.astype(np.uint8).copy(), user_selection, (0, 255, 0), cv.MARKER_CROSS, 10, 1)
		img2 = cv.drawMarker(img2.astype(np.uint8).copy(), user_selection, (0, 255, 0), cv.MARKER_CROSS, 10, 1)
		diff = cv.drawMarker(diff.astype(np.uint8).copy(), user_selection, (0, 255, 0), cv.MARKER_CROSS, 10, 1)
	blank = np.zeros((h, MARGIN, 3), dtype=np.uint8)
	# print("Viewing band #{} ({})".format(i, np.max(img2)))
	cv.imshow(WINDOW_NAME, np.concatenate((img1, blank, img2, blank, diff), axis=1))


def plot_bands(e, x, y, flages, param):
	global user_selection
	x = x % (w + MARGIN)
	user_selection = (x, y)
	view()
	if e == cv.EVENT_LBUTTONUP:
		subprocess.run(['python3', str(env.BASE / PROGRNAM_NAME), RESULT_PATH, '-p', "{},{}".format(x, y)])
		user_selection = None


cv.createTrackbar('select band', WINDOW_NAME , 0, d, view)
view(0)

cv.setMouseCallback(WINDOW_NAME, plot_bands)
cv.waitKey(0)

cv.destroyAllWindows()
cv.waitKey(10)

sys.exit(0)
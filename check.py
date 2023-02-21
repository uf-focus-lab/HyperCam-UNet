#!python
# Python Packages
import sys
from os.path import exists, basename
from tkinter.filedialog import askopenfilename
# PIP Packages
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# Custom Packages
import env
RESULT_PATH = askopenfilename(defaultextension=".npy", initialdir=str(env.RUN_PATH))
if RESULT_PATH is None:
	sys.exit(0)
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
		MARKER_SIZE = 16
		MARKER_WEIGHT = 1
		img1 = cv.drawMarker(img1.astype(np.uint8).copy(), user_selection, (  0,   0, 255), cv.MARKER_CROSS, MARKER_SIZE, MARKER_WEIGHT)
		img2 = cv.drawMarker(img2.astype(np.uint8).copy(), user_selection, (255, 128, 128), cv.MARKER_CROSS, MARKER_SIZE, MARKER_WEIGHT)
		diff = cv.drawMarker(diff.astype(np.uint8).copy(), user_selection, (  0, 255,   0), cv.MARKER_CROSS, MARKER_SIZE, MARKER_WEIGHT)
	blank = np.zeros((h, MARGIN, 3), dtype=np.uint8)
	# print("Viewing band #{} ({})".format(i, np.max(img2)))
	cv.imshow(WINDOW_NAME, np.concatenate((img1, blank, img2, blank, diff), axis=1))

band_plot = None
band_global_max = np.max([np.max(reference), np.max(result)])

def plot_bands(e, x, y, flags, param):
	global user_selection
	x = x % (w + MARGIN)
	user_selection = (x, y)
	view()
	# Plot on time axis
	global band_plot
	global band_global_max
	PLOT_WIN_NAME = 'Time Axis Plot'
	if (band_plot is None):
		band_plot = PLOT_WIN_NAME
		cv.namedWindow(PLOT_WIN_NAME, cv.WINDOW_AUTOSIZE)
		cv.startWindowThread()
	fig = plt.figure()
	line1 = np.reshape(reference[x, y, :], (-1))
	line2 = np.reshape(result[x, y, :], (-1))
	plt.plot(line1, 'r-')
	plt.plot(line2, 'b-')
	plt.ylim([0, band_global_max])
	fig.canvas.draw()
	fig_img = cv.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv.COLOR_RGBA2BGR)
	cv.imshow(PLOT_WIN_NAME, fig_img)

cv.createTrackbar('select band', WINDOW_NAME , 0, d, view)
view(0)

cv.setMouseCallback(WINDOW_NAME, plot_bands)
cv.waitKey(0)

cv.destroyAllWindows()
cv.waitKey(10)

sys.exit(0)
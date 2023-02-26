#!python
# Python Packages
import sys
from os.path import exists, dirname, basename, abspath
from tkinter.filedialog import askopenfilename
# PIP Packages
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
# Custom Packages
import env
from param import LED_LIST, REF_BANDS
CAPTURE_DIR = env.ensureDir(env.VAR_PATH / "capture")
isNpMtx = lambda f: type(f) == str and f.endswith(".npy") and exists(f)
askForMtx = lambda: askopenfilename(defaultextension=".npy", initialdir=str(env.RUN_PATH))
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

ID = basename(RESULT_PATH).replace(".npy", "")

prediction = np.load(RESULT_PATH)
h, w, d = prediction.shape

raw = np.load(str(env.DATA_PATH / (ID + ".npy")))
print(raw.dtype)
raw = np.stack([
	Image.fromarray(raw[:, :, d]).resize((w, h))
	for d in range(raw.shape[2])
], axis=2)
print(raw.shape)
reference = np.load(str(env.DATA_PATH / (ID + "_REF.npy")))
reference = (reference * 255 / np.max(reference)).astype(np.uint8)


WINDOW_NAME = "Inspecting {}".format(ID)
cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)
cv.startWindowThread()
cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_TOPMOST, 1)

MARGIN = 20
user_selection = None
i = 0
raw_display = np.stack([raw[:,:,_] for _ in [1,2,6]], axis=2).astype(np.uint8)

C = [
    ( 64,  64, 255),
    (200, 255,  64),
    ( 64, 255, 255),
    (255, 255, 255),
]
def TEXT(im, txt, pos=(10, 10), f=cv.FONT_HERSHEY_DUPLEX, scale=0.5, c=(255, 255, 255), b=1):
	return cv.putText(im.astype(np.uint8), txt, pos, f, scale, c, b, cv.LINE_AA)
def view(_i=None):
	global i
	if _i is not None:
		i = _i
	img1 = reference[:, :, i]
	img2 = prediction[:, :, i]
	# Compute diff image
	R = (img1 - img2) * (img1 > img2)
	B = (img2 - img1) * (img2 > img1)
	G = np.zeros(R.shape, dtype=np.uint8)
	diff = np.stack((B, G, R), axis=2)
	# Convert imgs to gray RGB
	img1, img2 = [np.stack((_, _, _), axis=2) for _ in [img1, img2]]
	# Put text
	img1 = TEXT(img1.astype(np.uint8).copy(), "Ground Truth", (10, h-10), c=C[0])
	_raw = TEXT(raw_display.copy(), "Raw data (RGB)", (10, h-10), c=C[1])
	img2 = TEXT(img2.astype(np.uint8).copy(), "Model Prediction", (10, h-10), c=C[2])
	diff = TEXT(diff.astype(np.uint8).copy(), "Difference", (10, h-10), c=C[3])
	if user_selection is not None:
		MARKER_SIZE = 16
		MARKER_WEIGHT = 1
		img1 = cv.drawMarker(img1, user_selection, C[0], cv.MARKER_CROSS, MARKER_SIZE, MARKER_WEIGHT)
		_raw = cv.drawMarker(_raw, user_selection, C[1], cv.MARKER_CROSS, MARKER_SIZE, MARKER_WEIGHT)
		img2 = cv.drawMarker(img2, user_selection, C[2], cv.MARKER_CROSS, MARKER_SIZE, MARKER_WEIGHT)
		diff = cv.drawMarker(diff, user_selection, C[3], cv.MARKER_CROSS, MARKER_SIZE, MARKER_WEIGHT)
	vBlank = 255 * np.ones((h, MARGIN, 3), dtype=np.uint8)
	row1 = np.concatenate((vBlank, img1, vBlank, _raw, vBlank), axis=1)
	row2 = np.concatenate((vBlank, img2, vBlank, diff, vBlank), axis=1)
	hBlank = 255 * np.ones((MARGIN, row1.shape[1], 3), dtype=np.uint8)
	display_img = np.concatenate((hBlank, row1, hBlank, row2, hBlank), axis=0)
	cv.imshow(WINDOW_NAME, display_img)
	return display_img

band_plot = None
global_max = [np.max(reference), np.max(raw), np.max(prediction)]
print(global_max)
PLT_TITLE = 'Spectrum of Selected Pixel'
fig = plt.figure()
plt.title(PLT_TITLE)
ax = fig.gca()
l_raw, = ax.plot([b for _,b,_ in LED_LIST], np.ones(len(LED_LIST)), 'cx')
l_ref, = ax.plot(REF_BANDS, np.zeros(d), 'r-')
l_prd, = ax.plot(REF_BANDS, np.zeros(d), 'b-')
l_raw.set_label("raw input")
l_ref.set_label("ground truth")
l_prd.set_label("prediction")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Intensity (relative)")
ax.set_xlim([390, 1000])
ax.set_ylim([0, 1])
ax.legend()

def plot_bands(e, x, y, flags, param):
	global user_selection
	x = x % (w + MARGIN) - MARGIN
	y = y % (h + MARGIN) - MARGIN
	if x < 0 or x >= w or y < 0 or y >= h:
		user_selection = None
	else:
		user_selection = (x, y)
	display_img = view()
	if user_selection is None:
		return
	# Plot on time axis
	global band_plot
	global band_global_max
	if (band_plot is None):
		band_plot = PLT_TITLE
		cv.namedWindow(PLT_TITLE, cv.WINDOW_AUTOSIZE)
		cv.startWindowThread()
	plt.figure(fig)
	l_ref.set_ydata(np.reshape(reference[y, x, :], (-1)) / global_max[0])
	l_raw.set_ydata(np.reshape(raw[y, x, :], (-1)) / global_max[1])
	l_prd.set_ydata(np.reshape(prediction[y, x, :], (-1)) / global_max[2])
	fig.canvas.draw()
	figure_img = cv.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv.COLOR_RGBA2BGR)
	cv.imshow(PLT_TITLE, figure_img)
	global i
	if e == cv.EVENT_LBUTTONUP:
		selection_bxy = [
			_[0] + str(_[1]) for _ in zip(['X', 'Y', 'B'], list(user_selection) + [i])]
		selection_xy = [_[0] + str(_[1]) for _ in zip(['X', 'Y'], list(user_selection))]
		file_path = CAPTURE_DIR / ('_'.join([CAPTURE_PREFIX] + selection_bxy) + '.png')
		cv.imwrite(str(file_path), display_img)
		file_path = CAPTURE_DIR / ('_'.join([CAPTURE_PREFIX] + selection_xy) + '.png')
		cv.imwrite(str(file_path), figure_img)
		print("Selection Saved => {}".format(file_path))

cv.createTrackbar('select band', WINDOW_NAME , 0, d-1, view)
view(0)

cv.setMouseCallback(WINDOW_NAME, plot_bands)
cv.waitKey(0)

cv.destroyAllWindows()
cv.waitKey(10)

sys.exit(0)
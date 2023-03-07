#!python
# Python Packages
import sys
from os import mkdir
from os.path import exists
from tkinter.messagebox import askyesnocancel
# PIP Packages
import cv2
import numpy as np
# Custom Packages
import env
from check_util.param import LED_LIST, REF_BANDS
# Components
from check_util.data import ID, RAW, REF, PRD, CAPTURE_DIR, CAPTURE_PREFIX
from check_util.view import view
from check_util.plot import plot

h, w, d = PRD.shape
# Interpolate spectral raw
gaussian = None
def raw(i):
	return np.stack([RAW[:, :, i] for i in [1, 2, 6]], axis=2)
# Generate differential image
def DIF(i):
	X, Y = REF[:, :, i], PRD[:, :, i]
	B, G, R = [np.zeros((h, w), dtype=np.uint8) for _ in range(3)]
	R = (X > Y) * (X - Y)
	B = (Y > X) * (Y - X)
	return np.stack([B, G, R], axis=2)

# Mouse callback
plot_list = []
def mouseCallback(e, x, y, flags, param):
	id, pos = update_cursor((x, y), e == cv2.EVENT_LBUTTONUP)
	if id is not None:
		plot_list.append((id, pos))
	if pos is not None:
		# Update band plot
		render_plot(id, pos)
		pass


LED_BANDS = [b for _, b, _ in LED_LIST]
render_plot = plot(
	"Pixel Spectrum",
	[
		(REF, ("Ground Truth", REF_BANDS, 'r-')),
		(PRD, ("Prediction"  , REF_BANDS, 'b-')),
		(RAW, ("Raw Input"   , LED_BANDS, 'cx')),
	]
)


rasterize, update_cursor, render = view(
	"Inspecting {}".format(CAPTURE_PREFIX),
	[
		(REF, "Ground Truth"          , ( 64,  64, 255)),
		(PRD, "Model Prediction"      , ( 64, 255, 255)),
		(raw, "Raw image (our camera)", (200, 255,  64)),
		(DIF, "Prediction error"      , (128, 255, 128)),
	],
	callback=mouseCallback
)

print("press ENTER to save selection")
k = cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(10)


if k != 13:
	print("key pressed", k)
	print("exiting")
	sys.exit(0)

DIR = env.ensureDir(CAPTURE_DIR / ID)
update_cursor(None)
grid_img, band_number = rasterize()
cv2.imwrite(str(DIR / "grid.png"), grid_img)
with open(DIR / "grid_wavelength.txt", "w") as f:
	f.write("{}nm\n".format(REF_BANDS[band_number]))

for id, pos in plot_list:
	plot_img = render_plot(id, pos)
	cv2.imwrite(str(DIR / (id + ".png")), plot_img)

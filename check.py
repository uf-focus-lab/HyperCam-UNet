#!python
# Python Packages
import sys
# PIP Packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Custom Packages
import env
from check_util.param import LED_LIST, REF_BANDS
# Components
from check_util.data import ARGS, ID, RAW, REF, PRD, CAPTURE, RUN_ID, CAPTURE_DIR, CAPTURE_PREFIX, TRAIN_LOG
from check_util.view import view, trim, text
from check_util.plot import plot

# Plot the training process and save to file
fig = plt.figure(figsize=(9, 4), dpi=160)
ax1 = fig.gca(); ax2 = ax1.twinx()
for key in TRAIN_LOG: TRAIN_LOG[key] = TRAIN_LOG[key][:200]
def getLim(*a, bleeding=0.05):
	a = np.array(a).reshape((-1,))
	r = (np.min(a), np.max(a))
	delta = (r[1] - r[0]) * bleeding
	return [r[0] - delta, r[1] + delta]
TRAIN_LOG["Loss"] = TRAIN_LOG["Loss"] / np.max(TRAIN_LOG["Loss"])
ax1.set_title("Training Process (Run ID = {})".format(RUN_ID))
ax1.set_ylim(getLim(TRAIN_LOG["Loss"]))
ax1.set_ylabel("Normalized Loss")
ax2.set_ylim(getLim(TRAIN_LOG["AvgErr"], TRAIN_LOG["StDev"]))
ax2.set_ylabel("Error")
line1, = ax1.plot(TRAIN_LOG["Epoch"], TRAIN_LOG["Loss" ], "r-")
line1.set_label("Loss")
line2, = ax2.plot(TRAIN_LOG["Epoch"], TRAIN_LOG["AvgErr"], "g--")
line2.set_label("Error - Average")
line3, = ax2.plot(TRAIN_LOG["Epoch"], TRAIN_LOG["StDev" ], "b--")
line3.set_label("Error - Std Dev")
lines = [line1, line2, line3]
ax1.legend(lines, [_.get_label() for _ in lines])
ax1.set_xlabel("Number of Epochs")
fig.canvas.draw()
buffer = np.asarray(fig.canvas.buffer_rgba())
buffer = trim(cv2.cvtColor(buffer, cv2.COLOR_RGBA2BGR))
cv2.imwrite(str(CAPTURE / RUN_ID) + '.png', buffer)

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
		# (PRD, ("Prediction"  , REF_BANDS, 'b-')),
		# (RAW, ("Raw Input"   , LED_BANDS, 'cx')),
	],
	size=[3, 5]
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

ENTER_KEY = 13
k = ENTER_KEY
if ARGS.points is None:
	print("Press ENTER to save selection")
	k = cv2.waitKey(0)
else:
	print("Using existing coordinates to plot bands")
	plot_list = []
	for c in ARGS.points.split(':'):
		l = c.split(',')
		assert len(l) == 2, f"Invalid coordinates <{c}>"
		x, y = [int(i) for i in l]
		plot_list.append(update_cursor((x, y), confirm=True, hasPadding=False))


if ARGS.band is not None:
	print(f"Rendering final result with band number {ARGS.band}")
	render(ARGS.band)
cv2.destroyAllWindows()
cv2.waitKey(10)


if k != ENTER_KEY:
	print("aborting...")
	sys.exit(0)


DIR = env.ensureDir(CAPTURE_DIR / ID)
update_cursor(None)
grid_img, band_number = rasterize()
cv2.imwrite(str(DIR / "grid.png"), grid_img)
with open(DIR / "grid_wavelength.txt", "w") as f:
	f.write("{}nm\n".format(REF_BANDS[band_number]))


imgs = []
for i, (id, pos) in zip(range(len(plot_list)), plot_list):
	img = render_plot(id, pos, draw_legend=(i + 1 == len(plot_list)))
	if i > 0:
		# img[190:-174, :158] = 255
		img[190:-150, :118] = 255
		img = img[:, 80:]
	if i + 1 < len(plot_list):
		img[190:-150, -80:] = 255
		img = img[:, :-40]
	imgs.append(img)

img = np.concatenate(imgs, axis=1)
cv2.imwrite(str(DIR / "bands.png"), img)

# Merge the results
SIZE = 1236
pad_top = 255 * np.ones((190, SIZE, 3), dtype=np.uint8)
pad_bottom = 255 * np.ones((174, SIZE, 3), dtype=np.uint8)

blank = np.ones((256, SIZE, 3), dtype=np.uint8) * 255
txt = trim(text(
	blank,
	"Point Selection, showing wavelength {}nm".format(REF_BANDS[band_number]),
	(0, 128), color=(0, 0, 0), scale=1.4, w=2
))
# Add text into blank space
h, w, _ = txt.shape
rx = [int((SIZE - w) / 2), None]
rx[1] = rx[0] + w
ry = [120 - h, 120]
pad_bottom[ry[0]:ry[1], rx[0]:rx[1]] = txt

grid_img = np.concatenate((pad_top, cv2.resize(grid_img, (SIZE, SIZE)), pad_bottom), axis=0)

merged_img = trim(np.concatenate(
	(grid_img, 255 * np.ones((img.shape[0], 20, 3), dtype=np.uint8), img),
	axis=1
))
cv2.imwrite(str(CAPTURE_DIR / (ID + ".png")), merged_img)
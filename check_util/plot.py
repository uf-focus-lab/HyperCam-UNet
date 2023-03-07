import matplotlib.pyplot as plt
import numpy as np
import cv2

def line_initializer(ax: plt.Axes):
	def init_line(attr):
		label, x, style = attr
		line = ax.plot(x, np.zeros(len(x)), style)[0]
		line.set_label(label)
		return line
	return init_line


def norm(arr: np.ndarray):
	arr = arr.astype(np.float64)
	arr = arr / np.stack((np.mean(arr, axis=2),), axis=2)
	arr = arr / np.stack((np.max(arr, axis=2),), axis=2)
	return arr


def plot(name, dataset, size=[4, 4], dpi=160):
	# unzip inputs
	imgs, attrs = zip(*dataset)
	imgs = list(map(norm, imgs))
	# Initialize plot
	fig = plt.figure(figsize=size, dpi=dpi)
	ax = fig.gca()
	# Initialize lines
	lines = list(map(line_initializer(ax), attrs))
	# Initialize axes
	ax.set_xlabel("Wavelength (nm)")
	ax.set_ylabel("Intensity (relative)")
	ax.set_xlim([390, 1000])
	ax.set_ylim([0, 1.1])
	ax.legend()
	# Initialize named window
	def render(id, pos):
		nonlocal ax, lines, imgs
		if id is not None:
			ax.set_title(
				"Spectrum of Pixel {} at ({}, {})".format(id, *pos)
			)
		else:
			ax.set_title(
				"Spectrum of Pixel ({}, {})".format(*pos)
			)
		for img, line in zip(imgs, lines):
			x, y = pos
			line.set_ydata(img[y, x, :])
		# Rasterize
		fig.canvas.draw()
		buffer = np.asarray(fig.canvas.buffer_rgba())
		buffer = cv2.cvtColor(buffer, cv2.COLOR_RGBA2BGR)
		cv2.imshow(name, buffer)
		return buffer
	# Start window thread
	cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
	cv2.startWindowThread()
	cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
	return render

# band_plot = None
# PLT_TITLE = 'Spectrum of Pixel A (x, y)'
# fig = plt.figure(figsize=(8,4), dpi=300)
# plt.title(PLT_TITLE)
# ax = fig.gca()
# fig.set_tight_layout()
# l_raw, = ax.plot([b for _,b,_ in LED_LIST], np.ones(len(LED_LIST)), 'cx')
# l_ref, = ax.plot(REF_BANDS, np.zeros(d), 'r-')
# l_prd, = ax.plot(REF_BANDS, np.zeros(d), 'b-')
# l_raw.set_label("raw input")
# l_ref.set_label("ground truth")
# l_prd.set_label("prediction")
# ax.set_xlabel("Wavelength (nm)")
# ax.set_ylabel("Intensity (relative)")
# ax.set_xlim([390, 1000])
# ax.set_ylim([0, 1.1])
# ax.legend()

# norm_raw = norm(RAW)
# norm_prd = norm(PRD)
# norm_ref = norm(REF)

# def plot_bands(e, x, y, flags, param):
# 	global user_selection
# 	x = x % (w + MARGIN)
# 	y = y % (h + MARGIN)
# 	if x < 0 or x >= w or y < 0 or y >= h:
# 		user_selection = None
# 	else:
# 		user_selection = (x, y)
# 	display_img = view()
# 	if user_selection is None:
# 		return
# 	# Plot on time axis
# 	global band_plot
# 	if (band_plot is None):
# 		band_plot = PLT_TITLE
# 		cv.namedWindow(PLT_TITLE, cv.WINDOW_AUTOSIZE)
# 		cv.startWindowThread()
# 	plt.figure(fig)
# 	l_ref.set_ydata(norm_ref[y, x, :])
# 	l_raw.set_ydata(norm_raw[y, x, :])
# 	l_prd.set_ydata(norm_prd[y, x, :])
# 	fig.canvas.draw()
# 	figure_img = cv.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv.COLOR_RGBA2BGR)
# 	cv.imshow(PLT_TITLE, figure_img)
# 	global i
# 	if e == cv.EVENT_LBUTTONUP:
# 		selection_bxy = [
# 			_[0] + str(_[1]) for _ in zip(['X', 'Y', 'B'], list(user_selection) + [i])]
# 		selection_xy = [_[0] + str(_[1]) for _ in zip(['X', 'Y'], list(user_selection))]
# 		file_path = CAPTURE_DIR / 'grid.png'
# 		cv.imwrite(str(file_path), display_img)
# 		file_path = CAPTURE_DIR / (selection_xy + '.png')
# 		cv.imwrite(str(file_path), figure_img)
# 		open(CAPTURE_DIR / 'grid_wavelength.txt', 'w').writelines([str(REF_BANDS[i])])
# 		print("Selection Saved => {}".format(file_path))

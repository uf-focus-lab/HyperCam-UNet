import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2

font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 11}

matplotlib.rc('font', **font)

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


def plot(name, dataset, size=[4, 5], dpi=160):
	# unzip inputs
	imgs, attrs = zip(*dataset)
	imgs = list(map(norm, imgs))
	# Initialize plot
	fig = plt.figure(figsize=size, dpi=dpi)
	ax = fig.gca()
	# Initialize lines
	lines = list(map(line_initializer(ax), attrs))
	# Initialize axes
	ax.set_xlabel("Wavelength (nm)", weight='bold')
	ax.set_ylabel("Intensity (relative)", weight='bold')
	ax.set_xlim([390, 1000])
	ax.set_ylim([0, 1.05])
	ax.grid(axis='y', which='major', dashes=(4, 4))
	legend = None
	# Initialize named window
	def render(id, pos, draw_legend=True):
		nonlocal ax, lines, imgs, legend
		if draw_legend: legend = ax.legend(loc="lower right", framealpha=0.8)
		elif legend is not None: legend.remove()
		if id is not None:
			ax.set_title(
				"Point {} ({}, {})".format(id, *pos), weight='bold'
			)
		else:
			ax.set_title(
				"Spectrum of Point ({}, {})".format(*pos), weight='bold'
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

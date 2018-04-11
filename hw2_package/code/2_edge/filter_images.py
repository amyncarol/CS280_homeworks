import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
from numpy.linalg import norm, inv
from matplotlib import cm
from math import pi, sqrt, exp
from mpl_toolkits.mplot3d import Axes3D
from math import cos, sin


def gaussian_kernel(sigma, truncate=4.0):
	"""
	return the gaussian distribution matrix given sigma and truncate.
	
	Args:
		simga: the standard deviation(int, number of pixels)
		truncate: where to truncate the distribution(truncate*sigma)
	
	Returns:
		a 2D numpy array of gaussian distribution(normalized)
	"""
	n = int(2*sigma*truncate + 1)
	gaussian = np.zeros((n, n))
	mean = np.array([n//2, n//2])
	for i in range(n):
		for j in range(n):
			gaussian[i, j] = 1/(2*pi*sigma**2)*exp(-((i-mean[0])**2+(j-mean[1])**2)/2/sigma**2)  
	return gaussian/gaussian.sum()

def elongated_gaussian_derivative(sigma1, sigma2, theta, truncate=4.0):
	"""
	return the 2D elongated_gaussian_derivative matrix given sigmas along each directions, orientation and truncate.
	
	Args:
		simga1: the standard deviation along the longer direction(int, number of pixels)
		simga2: the standard deviation along the shorter direction(int, number of pixels), take derivative only along this 
				direction
		theta: the orientation(in degree) of the longer direction, the shorter direction is normal to longer direction.
		truncate: where to truncate the distribution(truncate*sigma1)
	
	Returns:
		a 2D numpy array of elongated_gaussian_derivative(normalized)
	"""
	n = int(sigma1*truncate)
	x, y = np.meshgrid(np.arange(-n, n+1), np.arange(-n, n+1))
	orgpts = np.vstack((x.reshape(-1), y.reshape(-1)))

	theta = theta/180*pi
	rotation = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
	rotpts = rotation @ orgpts
 
	gx = np.exp(-1/2*(rotpts[0, :])**2/sigma1**2)
	gy = np.exp(-1/2*(rotpts[1, :])**2/sigma2**2) * (-(rotpts[1, :])/sigma2**2)
	egd = (gx * gy).reshape(2*n+1, 2*n+1)
	egd_plus = egd * (egd>0)
	return egd/egd_plus.sum()

def difference_filter(I):
	dx = np.array([[1, -1]])
	dy = np.array([[1], [-1]])

	return get_gradient_images(I, dx, dy)

def derivative_gaussian_filter(I, sigma):
	dx = np.array([[1, -1]])
	dy = np.array([[1], [-1]])
	dogx = signal.convolve2d(gaussian_kernel(sigma), dx, boundary='symm', mode='same')
	dogy = signal.convolve2d(gaussian_kernel(sigma), dy, boundary='symm', mode='same')
	plot_filter(dogx, 'derivative_gaussian_x.jpg')
	plot_filter(dogy, 'derivative_gaussian_y.jpg')

	return get_gradient_images(I, dogx, dogy)

def get_gradient_images(I, filterx, filtery):
	img_x = np.zeros(I.shape)
	img_y = np.zeros(I.shape)
	for i in range(3):
		img_x[:, :, i] = signal.convolve2d(I[:, :, i], filterx, boundary='symm', mode='same')
		img_y[:, :, i] = signal.convolve2d(I[:, :, i], filtery, boundary='symm', mode='same')
	img_x_norm = norm(img_x, axis = 2)
	img_y_norm = norm(img_y, axis = 2)

	img_mag = np.concatenate((img_x, img_y), axis = 2)
	img_mag = norm(img_mag, axis = 2)

	img_orient = np.arctan2(img_y, img_x).mean(axis=2)/pi*180
	return img_x_norm, img_y_norm, img_mag, img_orient

def oriented_filter(I):
	thetas = np.array([0, 30, 60, 90, 120, 150])
	sigma1s = 3*sqrt(2) ** np.array([1, 2, 3])
	theta, sigma1 = np.meshgrid(thetas, sigma1s)
	theta = theta.reshape(-1)
	sigma1 = sigma1.reshape(-1)

	img_mag = np.zeros((I.shape[0], I.shape[1], len(theta)))
	img_orient = np.zeros((I.shape[0], I.shape[1], len(theta)))

	for i in range(len(theta)):
		egd = elongated_gaussian_derivative(sigma1[i], sigma1[i]/3, theta[i])
		img_mag[:, :, i] = convolve3channel(I, egd)
		img_orient[:, :, i] = theta[i]

	img_max_repeat= np.repeat(img_mag.max(axis=2)[:, :, np.newaxis], len(theta), axis=2)
	indexes = np.equal(img_mag, img_max_repeat)
	img_orient = (img_orient * indexes).max(axis=2)

	img_mag = img_mag.max(axis=2)

	return img_mag, img_orient

def convolve3channel(I, filter_):
	filtered = np.zeros(I.shape)
	for i in range(3):
		filtered[:, :, i] = signal.convolve2d(I[:, :, i], filter_, boundary='symm', mode='same')
	filtered = norm(filtered, axis = 2)
	return filtered


def plots(img_mag, img_orient):
	fig, axes = plt.subplots(1, 2, figsize=(8, 4))
	#axes[0].imshow(img_x, cmap=cm.gray)
	#axes[0].set_title('diff x')

	#axes[1].imshow(img_y, cmap=cm.gray)
	#axes[1].set_title('diff y')

	axes[0].imshow(img_mag, cmap=cm.gray)
	axes[0].set_title('gradient magnitude')

	cax = axes[1].imshow(img_orient, cmap = cm.gist_rainbow)
	fig.colorbar(cax)
	axes[1].set_title('gradient orientation')
	return fig

def plot_filter(filter_, filename):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X = np.arange(0, filter_.shape[0], 1)
	Y = np.arange(0, filter_.shape[0], 1)
	X, Y = np.meshgrid(X, Y)
	surf = ax.plot_surface(X, Y, filter_, cmap=cm.coolwarm,
						   linewidth=0, antialiased=False)
	plt.savefig(filename)

def test_oriented_gaussian():
	for sigma1 in [10]:
		for theta in [0, 45, 90, 135]:
			plot_filter(elongated_gaussian_derivative(sigma1, sigma1/3, theta), 'tmp.png')
			plt.show()

def plot_1(folder):
	##finite difference
	img_x, img_y, img_mag, img_orient = difference_filter(image)
	for threshold in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
		threshold_ = threshold * img_mag.max()
		img_mag = img_mag * (img_mag > threshold_)
		img_orient = img_orient * (img_mag > threshold_)
		fig = plots(img_mag, img_orient)
		plt.savefig(os.path.join(folder, 'finite_threshold_{}.jpg'.format(threshold)))

def plot_2(folder):
	##derivative of gaussian
	img_x, img_y, img_mag, img_orient = derivative_gaussian_filter(image, sigma=2)
	for threshold in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
		threshold_ = threshold * img_mag.max()
		img_mag = img_mag * (img_mag > threshold_)
		img_orient = img_orient * (img_mag > threshold_)
		fig = plots(img_mag, img_orient)
		plt.savefig(os.path.join(folder, 'DoG_threshold_{}.jpg'.format(threshold)))

def plot_3(folder):
	####oriented gaussian derivative filters
	img_mag, img_orient = oriented_filter(image)
	for threshold in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
		threshold_ = threshold * img_mag.max()
		img_mag = img_mag * (img_mag > threshold_)
		img_orient = img_orient * (img_mag > threshold_)
		fig = plots(img_mag, img_orient)
		plt.savefig(os.path.join(folder, 'OF_threshold_{}.jpg'.format(threshold)))

parent = '/Users/yao/Desktop/CS280/hw2_package/2_edge'
for folder in range(2, 3):
	folder = os.path.join(parent, str(folder))
	print(folder) 
	img_file = os.listdir(folder)[0]
	print(img_file)
	image = plt.imread(os.path.join(folder, img_file))/255.0
	#plot_1(folder)
	#plot_2(folder)
	plot_3(folder)






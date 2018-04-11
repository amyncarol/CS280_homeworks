import matplotlib.pyplot as plt
from align_image_code import align_images
from scipy import signal
from math import pi, sqrt, exp
from skimage import io
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from skimage import color
import os

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

def LoG(sigma, truncate=4.0):
	lowpass = gaussian_kernel(sigma, truncate)
	highpass = np.zeros(lowpass.shape)
	i = lowpass.shape[0]//2
	highpass[i, i] = 1
	highpass = highpass - lowpass
	return highpass


def hybrid_image(im1, im2, sigma1, sigma2):
	"""
	takes two images and returns highpass filtered image, lowpass filtered images and the hybrid images,\
	the two images should be aligned.

	Args:
		im1: the image to display high frequency imformation, to be processed with high pass filter.
		im2: the image to display low frequency imformation, to be processed with low pass filter.
		sigma1: the cutoff for the highpass filter.
		sigma2: the cutoff for the lowpass filter.

	Returns:
		highpass filtered image, lowpass filter image and the hybrid image(in numpy 2D array).
	"""
	highpass = LoG(sigma = sigma1)
	im1_filter = np.zeros(im1.shape)
	for i in range(3):
		im1_filter[:, :, i] = signal.convolve2d(im1[:, :, i], highpass, boundary='symm', mode='same')
	im1_filter = (im1_filter-im1_filter.min())/(im1_filter.max()-im1_filter.min())

	lowpass2 = gaussian_kernel(sigma = sigma2)
	im2_filter = np.zeros(im2.shape)
	for i in range(3):
		im2_filter[:, :, i] = signal.convolve2d(im2[:, :, i], lowpass2, boundary='symm', mode='same')

	return im1_filter, im2_filter, im1_filter*0.5+im2_filter*0.5

def plot_filter(filter_, filename):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X = np.arange(0, filter_.shape[0], 1)
	Y = np.arange(0, filter_.shape[0], 1)
	X, Y = np.meshgrid(X, Y)
	surf = ax.plot_surface(X, Y, filter_, cmap=cm.coolwarm,
	                       linewidth=0, antialiased=False)
	plt.savefig(filename)


def plots(folder):
	# high sf
	im1 = plt.imread(os.path.join(folder, 'image1.jpg'))
	im1 = im1[:,:,:3]

	# low sf
	im2 = plt.imread(os.path.join(folder, 'image2.jpg'))
	im2 = im2[:,:,:3]

	# Next align images (this code is provided, but may be improved)
	im1_aligned, im2_aligned = align_images(im1, im2)

	## You will provide the code below. Sigma1 and sigma2 are arbitrary 
	## cutoff values for the high and low frequencies
	sigma1 = 2
	sigma2 = 3

	im1_filtered, im2_filtered, hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

	### plot the hybrid image
	plt.imshow(hybrid)
	plt.savefig(os.path.join(folder,'hybrid_'+str(sigma1)+'_'+str(sigma2)+'.png'))
	plt.imshow(im1_filtered)
	plt.savefig(os.path.join(folder,'im1_filtered_'+str(sigma1)+'_'+str(sigma2)+'.png'))
	plt.imshow(im2_filtered)
	plt.savefig(os.path.join(folder,'im2_filtered_'+str(sigma1)+'_'+str(sigma2)+'.png'))


	#im1_filtered, im2_filtered, hybrid = hybrid_image(im1, im1, sigma1, sigma2)
	###show fft of images#####
	#fig, axes = plt.subplots(3, 2, figsize=(2*4, 3*4))
	#axes[0, 0].imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(color.rgb2gray(im1))))), cmap=cm.gray)
	#axes[0, 0].set_title('image1')
	#axes[0, 1].imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(color.rgb2gray(im1_filtered))))),  cmap=cm.gray)
	#axes[0, 1].set_title('highpass of image1, sigma='+str(sigma1))
	#axes[1, 0].imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(color.rgb2gray(im2))))), cmap=cm.gray)
	#axes[1, 0].set_title('image2')
	#axes[1, 1].imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(color.rgb2gray(im2_filtered))))),  cmap=cm.gray)
	#axes[1, 1].set_title('lowpass of image2, sigma='+str(sigma2))
	#axes[2, 1].imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(color.rgb2gray(hybrid))))),  cmap=cm.gray)
	#axes[2, 1].set_title('hybrid')
	#plt.savefig(os.path.join(folder,'fft_'+str(sigma1)+'_'+str(sigma2)+'.jpg'), bbox_inches='tight')

	###plot the highpass and lowpass filter
	#highpass = LoG(sigma1)
	#plot_filter(highpass, os.path.join(folder,'highpass.png'))
	#lowpass = gaussian_kernel(sigma2)
	#plot_filter(lowpass, os.path.join(folder,'lowpass.png'))


plots('wukong')







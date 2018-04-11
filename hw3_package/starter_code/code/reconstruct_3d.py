import numpy as np
import cv2
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from my_functions import *
#from mpl_toolkits.mplot3d import Axes3D
#from IPython import embed

# """
# Homework 2: 3D reconstruction from two Views
# This function takes as input the name of the image pairs (i.e. 'house' or
# 'library') and returns the 3D points as well as the camera matrices...but
# some functions are missing.
# NOTES
# (1) The code has been written so that it can be easily understood. It has 
# not been written for efficiency.
# (2) Don't make changes to this main function since I will run my
# reconstruct_3d.m and not yours. I only want from you the missing
# functions and they should be able to run without crashing with my
# reconstruct_3d.m
# (3) Keep the names of the missing functions as they are defined here,
# otherwise things will crash
# """

VISUALIZE = False

def reconstruct_3d(name):
	# ------- Load images, K matrices and matches -----
	data_dir = "../data/{}".format(name)

	# images
	I1 = cv2.imread(f"{data_dir}/{name}1.jpg")
	I2 = cv2.imread(f"{data_dir}/{name}2.jpg")
	# of shape (H,W,C)

	# K matrices
	K1 = scipy.io.loadmat(f"{data_dir}/{name}1_K.mat")["K"]
	K2 = scipy.io.loadmat(f"{data_dir}/{name}2_K.mat")["K"]

	# corresponding points
	lines = open(f"{data_dir}/{name}_matches.txt").readlines()
	matches = np.array([list(map(float, line.split())) for line in lines])

	# this is a N x 4 where:
	# matches(i,1:2) is a point (w,h) in the first image
	# matches(i,3:4) is the corresponding point in the second image

	if VISUALIZE:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt.imshow(np.concatenate([I1, I2], axis=1))
		plt.plot(matches[:, 0], matches[:, 1], "+r")
		plt.plot(matches[:, 2] + I1.shape[1], matches[:, 3], "+r")
		for i in range(matches.shape[0]):
			line = Line2D([matches[i, 0], matches[i, 2] + I1.shape[1]], [matches[i, 1], matches[i, 3]], linewidth=1,
						  color="r")
			#ax.add_line(line)
		plt.savefig(f"{data_dir}/{name}_matches.jpg")
		#plt.show()

	## -------------------------------------------------------------------------
	## --------- Find fundamental matrix --------------------------------------

	# F        : the 3x3 fundamental matrix,
	# res_err  : mean squared distance between points in the two images and their
	# their corresponding epipolar lines

	(F, res_err) = fundamental_matrix(matches)  # <------------------------------------- You write this one!
	print(f"Residual in F = {res_err}")
	print('the fundamental matrix F is {}'.format(F))
	E = K2.T @ F @ K1

	## -------------------------------------------------------------------------
	## ---------- Rotation and translation of camera 2 ------------------------

	# R : cell array with the possible rotation matrices of second camera
	# t : cell array of the possible translation vectors of second camera
	(R, t) = find_rotation_translation(E)  # <------------------------------------- You write this one!

	print(t)
	print(R)


	# Find R2 and t2 from R,t such that largest number of points lie in front
	# of the image planes of the two cameras
	P1 = K1 @ np.concatenate([np.identity(3), np.zeros((3, 1))], axis=1)

	# the number of points in front of the image planes for all combinations
	num_points = np.zeros([len(t), len(R)])
	errs = np.full([len(t), len(R)], np.inf)

	for ti in range(len(t)):
		t2 = t[ti]
		for ri in range(len(R)):
			R2 = R[ri]
			P2 = K2 @ np.concatenate([R2, t2[:, np.newaxis]], axis=1)
			(points_3d, errs[ti,ri]) = find_3d_points(P1, P2, matches) #<---------------------- You write this one!
			Z1 = points_3d[:,2]
			Z2 = (points_3d @ R2[2,:].T + t2[2])
			num_points[ti,ri] = np.sum(np.logical_and(Z1>0,Z2>0))
	(ti,ri) = np.where(num_points==np.max(num_points))
	j = 0 # pick one out the best combinations
	print(f"Reconstruction error = {errs[ti[j],ri[j]]}")

	t2 = t[ti[j]]
	R2 = R[ri[j]]

	# print(num_points)
	# print(errs)
	# print(t2)
	# print(R2)

	P2 = K2 @ np.concatenate([R2, t2[:, np.newaxis]], axis=1)

	# % compute the 3D points with the final P2
	points, _, matches_hat = find_3d_points(P1, P2, matches, True) # <---------------------------------------------- You have already written this one!

	## -------- plot points and centers of cameras ----------------------------

	plot_3d(points, t2, R2) #<-------------------------------------------------------------- You write this one!
	#visualize_matches(I1, I2, matches_hat)

if __name__ == "__main__":
	reconstruct_3d('library')
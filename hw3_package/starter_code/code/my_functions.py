from numpy.linalg import norm, svd, det, matrix_rank
import numpy as np
from numpy import apply_along_axis
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
def fundamental_matrix(matches):
	"""
	calculate the fundamental matrix given matching points from two images, also calculate residual error

	Parameters:

		matches: # this is a N x 4 where:
				 # matches(i,1:2) is a point (w,h) in the first image
				 # matches(i,3:4) is the corresponding point in the second image

	 Returns: (F, res_err)

		# F        : the 3x3 fundamental matrix,
		# res_err  : mean squared distance between points in the two images and their
		# their corresponding epipolar lines

	"""
	###normalization
	T1, image1 = normalization(matches[:, :2])
	T2, image2 = normalization(matches[:, 2:])

	##svd
	A = get_A(image1, image2)
	u, s, vh = svd(A)
	f = vh.T[:, -1]
	F = f.reshape(3, 3)

	##force rank 2
	u, s, vh = svd(F)
	s[2] = 0
	s = np.diag(s)
	F = u @ s @ vh

	##denormalization
	F = T2.T @ F @ T1

	res_err, objective = get_res(F, matches)

	print('The optimization objective is {}'.format(objective))

	return (F, res_err)

def normalization(image):
	"""
	Normalization: the mean of the points is at the origin, and scaling them by 
					   sigma (so that the mean distance from the origin is a constant (e.g. sqrt(2))

	Parameters: 
		image: this is a N x 2 of (x, y) positions in an image

	Returns: T, image_normalized
		T: the transformation matrix of size (3, 3)
		image_normalized: this is (x, y) positions normalized in an image, size (N, 2)
	"""

	mean = np.mean(image, axis = 0)
	image_shifted = image - mean
	sigma = np.mean(norm(image_shifted, axis=1))
	T = 1/sigma * np.array([[1, 0, -mean[0]], [0, 1, -mean[1]], [0, 0, sigma]])

	n = image.shape[0]
	image_homo = np.hstack((image, np.ones((n, 1))))
	return T, (image_homo @ T.T)[:, :2]

def get_A(image1, image2):
	"""
	given a (N, 2) N points matrix, get its corresponding A matrix of size (N, 9)

	Paramters: 
		image1, image2: two (N, 2) matrices of N corresponding points

	Returns: 
		A: (N, 9) as defined in CS280-hw3
	"""
	def get_a(point):
		x1 = point[0]
		y1 = point[1]
		x2 = point[2]
		y2 = point[3]
		return np.array([x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1])

	image = np.hstack((image1, image2))
	return apply_along_axis(get_a, 1, image)

def get_res(F, matches):
	"""
	given the fundamental matrix F of size (3, 3) and 2 sets of corresponding points, return the residual error
	
	Args:
		F: the fundamental matrix of size (3, 3)
		matches: # this is a N x 4 where:
				 # matches(i,1:2) is a point (w,h) in the first image
				 # matches(i,3:4) is the corresponding point in the second image
	Returns: 
		the residual error and the objective (objective should be close to zero)
	"""
	res = 0
	objective = 0
	for match in matches:
		x1 = np.array([[match[0], match[1], 1]]).T
		x2 = np.array([[match[2], match[3], 1]]).T
		res += (x2.T @ F @ x1)**2/2 * (1/norm(F @ x1)**2 + 1/norm(F.T @ x2)**2)
		objective += (x2.T @ F @ x1)**2
	return res/matches.shape[0], objective/matches.shape[0]

def find_rotation_translation(E):
	"""
	given essential matrix, return Rotation and translation

	Args:
		E: essential matrix
	Returns:
		(R, t)
		R : cell array with the possible rotation matrices of second camera
		t : cell array of the possible translation vectors of second camera
	"""
	u, s, vh = svd(E)
	t = u[:, 2]
	t_list = [t, -t]

	R1 = u @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ vh
	R2 = u @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).T @ vh

	R_list = []
	for R in [R1, R2, -R1, -R2]:
		if det(R) > 0:
			R_list.append(R)

	return R_list, t_list

def find_3d_points(P1, P2, matches, cal_matches_hat=False):
	"""
	given two carema matrix P1, P2 and the matches points, solve for the 3D points. Also calculate reprojection error

	Args:
		P1, P2: the camera matrix
		matches: as defined before
		cal_matches_hat: whether return matches_hat(reprojection)

	Returns: (points_3d, rec_err)
		points_3d: a matrix of (N, 3)
		rec_err: the reprojection error
	"""
	n = matches.shape[0]
	points_3d = np.zeros((n, 3))
	rec_err = 0
	matches_hat = np.zeros_like(matches)
	for i in range(n):
		B1 = construct_B(P1, [matches[i][0], matches[i][1]])
		B2 = construct_B(P2, [matches[i][2], matches[i][3]])
		B = np.vstack((B1, B2))

		u, s, vh = svd(B)
		# for j, value in enumerate(s):
		#   print(value)
		# print('\n')

		point_3d = vh.T[:, -1]
		points_3d[i, :] = point_3d[:3]/point_3d[3]

		##calculate reprojection error
		x1 = np.array([matches[i][0], matches[i][1]]).reshape((2, 1))
		x2 = np.array([matches[i][2], matches[i][3]]).reshape((2, 1))
		x1_hat = P1 @ point_3d.reshape((4, 1))
		x1_hat = x1_hat[:2]/x1_hat[2]
		x2_hat = P2 @ point_3d.reshape((4, 1))
		x2_hat = x2_hat[:2]/x2_hat[2]
		# print(x1)
		# print(x1_hat)
		matches_hat[i, :] = np.hstack((x1_hat.reshape((1, 2)), x2_hat.reshape((1, 2))))

		rec_err += 1/2 * (norm(x1-x1_hat)**2 + norm(x2-x2_hat)**2)

	if cal_matches_hat:
		return points_3d, rec_err/n, matches_hat
	else:
		return points_3d, rec_err/n


def construct_B(P, point):
	row1 = np.array([[P[2, 0]*point[0]-P[0, 0], P[2, 1]*point[0]-P[0, 1], P[2, 2]*point[0]-P[0, 2], P[2, 3]*point[0]-P[0, 3]]])
	row2 =  np.array([[P[2, 0]*point[1]-P[1, 0], P[2, 1]*point[1]-P[1, 1], P[2, 2]*point[1]-P[1, 2], P[2, 3]*point[1]-P[1, 3]]])
	return np.vstack((row1, row2))

def choose_best_t_R(t, R, matches):
	"""
	this is just a copy of part of the code in reconstruct_3d.py that chooses the best t and R

	Algorithm: Find R and t such that largest number of points lie in front of the image planes of the two cameras

	Args:
		t:  contains possible translations
		R: contains possible rotations
		matches: as defined above
	Returns:
		(t2, R2)
		the best translation and rotation
	"""
	P1 = np.concatenate([np.identity(3), np.zeros((3, 1))], axis=1)

	num_points = np.zeros([len(t), len(R)])
	errs = np.full([len(t), len(R)], np.inf)

	for ti in range(len(t)):
		t2 = t[ti]
		for ri in range(len(R)):
			R2 = R[ri]
			P2 = np.concatenate([R2, t2[:, np.newaxis]], axis=1)
			(points_3d, errs[ti,ri]) = find_3d_points(P1, P2, matches)
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

	return (t2, R2)

def plot_3d(points_3d, t, R):
	"""
	3D plot of the 3D scene points and the location of the two camera centers

	Args:
		points_3d: (N, 3) matrix of the 3D scene points
		t: the translation vector for the second camera, the first is at (0, 0, 0)
		R: the rotation vector for the second camera
	"""
	C = -R.T @ t.reshape((3, 1))
	C = C.reshape((3))

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='skyblue', s=30)

	cameras = np.array([[0, 0, 0], [C[0], C[1], C[2]]])
	ax.scatter(cameras[0, 0], cameras[0, 1], cameras[0, 2], c='red', s=70, label = 'camera1')
	ax.scatter(cameras[1, 0], cameras[1, 1], cameras[1, 2], c='green', s=70, label = 'camera2')
	
	all_points = np.vstack((points_3d, cameras))
	X = all_points[:, 0]
	Y = all_points[:, 1]
	Z = all_points[:, 2]
	max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
	mid_x = (X.max()+X.min()) * 0.5
	mid_y = (Y.max()+Y.min()) * 0.5
	mid_z = (Z.max()+Z.min()) * 0.5
	ax.set_xlim(mid_x - max_range, mid_x + max_range)
	ax.set_ylim(mid_y - max_range, mid_y + max_range)
	ax.set_zlim(mid_z - max_range, mid_z + max_range)

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	plt.legend()
	ax.view_init(-90, -90)
	plt.show()

def visualize_matches(I1, I2, matches):
	"""
	visualize matches on the two images
	Args:
		I1, I2: two images
		matches: the (N, 4) matrix
	"""
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.imshow(np.concatenate([I1, I2], axis=1))
	plt.plot(matches[:, 0], matches[:, 1], "+r")
	plt.plot(matches[:, 2] + I1.shape[1], matches[:, 3], "+r")
	for i in range(matches.shape[0]):
		line = Line2D([matches[i, 0], matches[i, 2] + I1.shape[1]], [matches[i, 1], matches[i, 3]], linewidth=1,
					  color="r")
		#ax.add_line(line)
	plt.title("reprojection")
	plt.show()

# if __name__ == "__main__":
#   # a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
#   # print(a.reshape(3, 3))

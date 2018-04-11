import unittest
from numpy.testing import assert_array_equal, assert_almost_equal
import numpy as np
from my_functions import *
from math import sqrt
import matplotlib.pyplot as plt
from numpy.linalg import svd

class TestMyFunctions(unittest.TestCase):
	def setUp(self):
		points_3d = np.array([[-0.5, -0.5, 8], [0.5, -0.5, 8], [-0.5, 0.5, 8], [-0.5, -0.5, 9], 
			[-0.5, 0.5, 9], [0.5, -0.5, 9], [0.5, 0.5, 8], [0.5, 0.5, 9], [0.5, 1, 9]])
		a = 2
		t = np.array([a, 0, 0])
		t_cross = np.array([[0, 0, 0], [0, 0, -a], [0, a, 0]])
		#R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
		R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
		E = t_cross @ R

		P1 = np.concatenate([np.identity(3), np.zeros((3, 1))], axis=1)
		P2 = np.concatenate([R, t[:, np.newaxis]], axis=1)

		n = points_3d.shape[0]
		matches = np.zeros((n, 4))
		point_3d_homo = np.hstack((points_3d, np.ones((n, 1))))
		#print(point_3d_homo)
		for i in range(n):
			x1 = P1 @ point_3d_homo[i, :].reshape((4, 1))
			x2 = P2 @ point_3d_homo[i, :].reshape((4, 1))
			matches[i, 0] = x1[0]/x1[2]
			matches[i, 1] = x1[1]/x1[2]
			matches[i, 2] = x2[0]/x2[2]
			matches[i, 3] = x2[1]/x2[2]
		#print(matches)
		# plt.plot(matches[:, 0], matches[:, 1], 'o')
		# plt.show()
		# plt.plot(matches[:, 2], matches[:, 3], 'o')
		# plt.show()

		self.t = t
		self.R = R
		self.E = E
		self.P1 = P1
		self.P2 = P2
		self.matches = matches
		self.points_3d = points_3d


	def test_normalization(self):
		image = np.array([[1, 3], [3, 1]])
		T, image_normalized = normalization(image)
		assert_almost_equal(T, np.array([[1/sqrt(2), 0, -sqrt(2)], [0, 1/sqrt(2), -sqrt(2)], [0, 0, 1]]))
		assert_almost_equal(image_normalized, np.array([[-1/sqrt(2), 1/sqrt(2)], [1/sqrt(2), -1/sqrt(2)]]))

	def test_get_A(self):
		image1 = np.array([[1, 2], [3, 4]])
		image2 = np.array([[0, 1], [1, 0]])
		A = get_A(image1, image2)
		assert_array_equal(A, np.array([[0,0,0,1,2,1,1,2,1],[3,4,1,0,0,0,3,4,1]]))

	def test_find_rotation_translation(self):
		R_list, t_list = find_rotation_translation(self.E)
		# print(t_list)
		# print(R_list)

	def test_fundamental_matrix(self):
		F, res_err = fundamental_matrix(self.matches)
		assert_almost_equal(np.absolute(self.E/norm(self.E)), np.absolute(F/norm(F)), decimal=3)

	def test_find_3d_points(self):
		points_3d_hat, rec_err, matches_hat = find_3d_points(self.P1, self.P2, self.matches, True)
		assert_almost_equal(self.points_3d, points_3d_hat)
		assert_almost_equal(0, rec_err)
		assert_almost_equal(self.matches, matches_hat)

	def test_choose_best_t_R(self):
		(F, res_err) = fundamental_matrix(self.matches)
		(R, t) = find_rotation_translation(F)
		# print(self.E)
		# print(F)
		(t2, R2) = choose_best_t_R(t, R, self.matches)
		assert_almost_equal(t2/norm(t2), self.t/norm(self.t))
		assert_almost_equal(R2, self.R, decimal=2)

	def test_plot_3d_original(self):
		#plot_3d(self.points_3d, self.t, self.R)
		pass

	def test_plot_3d_derived(self):
		(F, res_err) = fundamental_matrix(self.matches)
		(R, t) = find_rotation_translation(F)
		(t2, R2) = choose_best_t_R(t, R, self.matches)
		P1 = np.concatenate([np.identity(3), np.zeros((3, 1))], axis=1)
		P2 = np.concatenate([R2, t2[:, np.newaxis]], axis=1)
		points_3d_hat, rec_err, matches_hat = find_3d_points(P1, P2, self.matches, True)
		assert_almost_equal(self.matches, matches_hat, decimal=2)
		#plot_3d(points_3d_hat, t2, R2)
		pass


if __name__ == '__main__':
	unittest.main()
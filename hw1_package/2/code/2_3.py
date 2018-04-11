
# coding: utf-8

# In[35]:


from random import random
import numpy as np
from math import pi, sin, cos
def get_unit_vector():
    s = np.array([random(), random(), random()])
    s = s/np.linalg.norm(s)
    return s

def get_s_matrix(s):
    return np.array([[0, -s[2], s[1]], [s[2], 0, -s[0]], [-s[1], s[0], 0]])

def get_rotation_matrix(s, phi):
    S = get_s_matrix(s)
    R = np.eye(3) + sin(phi) * S + (1-cos(phi)) * S @ S
    return R
    
def get_p_rotation(s, phi, p):
    R = get_rotation_matrix(s, phi)
    return np.dot(R, p)

def plot_3d(s, p, phi, fig):
    ax = fig.gca(projection='3d')
    x = np.linspace(0, s[0], 100)
    y = np.linspace(0, s[1], 100)
    z = np.linspace(0, s[2], 100)
    ax.plot(x, y, z)
    
    n = phi.shape[0]
    px = np.zeros(n)
    py = np.zeros(n)
    pz = np.zeros(n)
    for i, angle in enumerate(phi):
        new_p = get_p_rotation(s, angle, p)
        px[i] = new_p[0]
        py[i] = new_p[1]
        pz[i] = new_p[2]
    ax.scatter(px, py, pz)
    return ax

s = get_unit_vector()
p = get_unit_vector()
phi = np.array([0, pi/12, pi/8, pi/6, pi/4, pi/2, pi, 3*pi/2])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = plot_3d(s, p, phi, fig)
plt.xlim([-1,1])
plt.ylim([-1,1])
ax.set_zlim(-1,1)
plt.show()



# In[13]:





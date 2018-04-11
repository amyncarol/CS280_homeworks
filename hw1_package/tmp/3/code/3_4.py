
# coding: utf-8

# In[49]:


import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from PIL import Image


def get_A_homo(u, v):
    return np.array([[u[0], u[1], 1, 0, 0, 0, -u[0]*v[0], -u[1]*v[0]],                     [0, 0, 0, u[0], u[1], 1, -u[0]*v[1], -u[1]*v[1]]])
   
def get_A_affine(u, v):
    return np.array([[u[0], u[1], 1, 0, 0, 0],                     [0, 0, 0, u[0], u[1], 1]])

def affine_solve(U, V):
    b = np.ravel(V, order='F')
    b = b.reshape((b.shape[0], 1))
    A = get_A_affine(U[:, 0], V[:, 0])
    for i in range(1, U.shape[1]):
        A = np.vstack((A, get_A_affine(U[:, i], V[:, i])))
    h = inv(A.T @ A) @ A.T @ b
    H = np.vstack((h[:3, :].T, h[3:6, :].T, [[0, 0, 1]]))
    return H

def homography_solve(U, V):
    b = np.ravel(V, order='F')
    b = b.reshape((b.shape[0], 1))
    A = get_A_homo(U[:, 0], V[:, 0])
    for i in range(1, U.shape[1]):
        A = np.vstack((A, get_A_homo(U[:, i], V[:, i])))
    h = inv(A.T @ A) @ A.T @ b
    H = np.vstack((h[:3, :].T, h[3:6, :].T, [[h[6, 0], h[7, 0], 1]]))
    return H

def homography_transform(u, H):
    N = u.shape[1]
    u = np.vstack((u, np.ones((1, N))))
    V_target = H @ u
    V_target = from_3D_to_2D(V_target)
    V_target =  V_target.astype(int)
    return V_target

def from_3D_to_2D(V):
    N = V.shape[1]
    V = V @ inv(np.diag(V[2, 0:N]))
    return V[0:2, :]
     
def get_U_matrix(U):
    minx = min(U[0, :])
    maxx = max(U[0, :])
    miny = min(U[1, :])
    maxy = max(U[1, :])
    x_length = maxx-minx
    y_length = maxy-miny
    print(x_length)
    print(y_length)
    vx = np.arange(minx, maxx)
    vy = np.arange(miny, maxy)
    U_source = np.vstack((np.tile(vx, y_length), np.ravel(np.tile(vy, (x_length, 1)).T)))
    return U_source
            
def get_transform(U, V, transform):
    U_source = get_U_matrix(U)
    if transform == 'affine':
        H = affine_solve(U, V)
    elif transform == 'homography':
        H = homography_solve(U, V)
    return homography_transform(U_source, H)

def superimpose(U, V, source_img, target_img, transform):
    x = sum(U)[2]
    y = sum(U)[0]
    print(x, y)
    U_source = get_U_matrix(U)
    V_target = get_transform(U, V, transform)
    print(U_source.shape)
    print(U_source)
    print(V_target)
    for i in range(U_source.shape[1]):
        #target_img[V_target[1, i], V_target[0, i], :] = source_img[U_source[1, i], U_source[0, i], :]
        if i%x != x-1 and int(i)//int(x) != y-1:
            j = i + x + 1
            x_inc = max(V_target[0, j], V_target[0, i+1]) - V_target[0, i]
            y_inc = max(V_target[1, j], V_target[1, i+x]) - V_target[1, i]
            for m in range(0, x_inc+1):
                for n in range(0, y_inc+1):
                    target_img[V_target[1, i]+n, V_target[0, i]+m, :] = source_img[U_source[1, i], U_source[0, i], :]
    return target_img

def show_img(file, rotation = 0, size = None):
    img = Image.open(file)
    if size != None:
        img.thumbnail((size, size), Image.ANTIALIAS) # resizes image in-place
    img = ndimage.rotate(img, rotation)
    #imgplot = plt.imshow(img)
    #plt.show()
    return img


# In[ ]:

# target_img = show_img('images/times_square.jpg')
# V_list = [np.array([[802, 894, 892, 808],[608, 609, 494, 493]]),
#           np.array([[795, 903, 897, 803], [769, 766, 624, 621]]),
#           np.array([[154, 279, 402, 311], [152, 316, 149, 15]]),
#           np.array([[530, 575, 617, 582], [281, 353, 236, 161]]),
#           np.array([[279, 370, 431, 364], [863, 883, 725, 672]]),
#           np.array([[1173, 1224, 1191, 1148], [797,769, 661, 701]]), 
#           np.array([[473, 560, 577, 506], [755, 820, 743, 651]])]

# #V_list = [np.array([[858, 1043, 1038, 950], [302, 390, 234, 236]])]

# for V in V_list:
#     y_length = V[1, 0] - V[1, 3]
#     source_img = show_img('images/guai.jpg', 0, int(y_length/3))
#     y = source_img.shape[0]
#     x = source_img.shape[1]
#     U = np.array([[0, x, x, 0], [y, y, 0, 0]])
#     target_img = superimpose(U, V, source_img, target_img, transform = 'homography') 
#     fig, ax = plt.subplots()
#     plt.imshow(target_img)
#     plt.axis('off')

# extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
# plt.savefig('images/time_square_guai.jpg', dpi=300, format='jpg', bbox_inches=extent)
# plt.show()

# In[ ]:


target_img = show_img('images/doe.jpg')
V_list = [np.array([[42,132, 141, 57], [339,352, 95, 25]]),
          np.array([[166, 215, 220, 172], [356, 362, 162, 121]]), 
          np.array([[236, 267, 271, 240], [367, 371, 205, 179]]),
          np.array([[280, 304, 306, 284], [373, 376, 234, 217]]),
          np.array([[349, 395, 395, 350], [362, 363, 273, 273]]),
          np.array([[439, 569, 569, 439], [377, 375, 231, 231]]),
          np.array([[612, 658, 656, 612], [360, 360, 273, 273]])]
          

#V_list = [np.array([[858, 1043, 1038, 950], [302, 390, 234, 236]])]

for V in V_list:
    y_length = V[1, 0] - V[1, 3]
    source_img = show_img('images/guai.jpg', 0, int(y_length/3))
    y = source_img.shape[0]
    x = source_img.shape[1]
    U = np.array([[0, x, x, 0], [y, y, 0, 0]])
    target_img = superimpose(U, V, source_img, target_img, transform = 'affine') 
    fig, ax = plt.subplots()
    plt.imshow(target_img)
    plt.axis('off')

extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plt.savefig('images/doe_guai.jpg', dpi=300, format='jpg', bbox_inches=extent)
plt.show()


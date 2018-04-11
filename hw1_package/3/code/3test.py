
# coding: utf-8

# In[25]:
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from PIL import Image

def affine_solve(U, V):
    N = U.shape[1]
    b = np.ones((N, 1))
    H22 = (N * V @ U.T - V @ b @ b.T @ U.T) @ inv(N * U @ U.T - U @ b @ b.T @ U.T)
    H21 = (V - H22 @ U) @ b @ inv(b.T @ b)
    H12 = np.zeros((1, 2))
    H11 = np.ones((1, 1))
    return np.vstack((np.hstack((H22, H21)), np.hstack((H12, H11))))

def from_3D_to_2D(V):
    N = V.shape[1]
    V = V @ inv(np.diag(V[2, 0:N]))
    return V[0:2, :]
     
def get_U_matrix(U):
    x = sum(U)[2]
    y = sum(U)[0]
    vx = np.arange(0, x)
    vy = np.arange(0, y)
    U_source = np.vstack((np.tile(vx, y), np.ravel(np.tile(vy, (x, 1)).T)))
    return U_source

def get_transform(U, V):
    U_source = get_U_matrix(U)
    H = affine_solve(U, V)
    N = U_source.shape[1]
    U_source = np.vstack((U_source, np.ones((1, N))))
    V_target = H @ U_source
    V_target = from_3D_to_2D(V_target)
    return V_target.astype(int)
  

def superimpose(U, V, source_img, target_img):
    x = sum(U)[2]
    y = sum(U)[0]
    print(x, y)
    U_source = get_U_matrix(U)
    V_target = get_transform(U, V)
    print(U_source.shape)
    print(U_source)
    print(V_target)
    for i in range(U_source.shape[1]):
        #target_img[V_target[1, i], V_target[0, i], :] = source_img[U_source[1, i], U_source[0, i], :]
        if i%x != x-1 and int(i)//int(x) != y-1:
            j = i + x + 1
            x_inc = V_target[0, j] - V_target[0, i]
            y_inc = V_target[1, j] - V_target[1, i]
            for m in range(0, x_inc+1):
                for n in range(0, y_inc+1):
                    target_img[V_target[1, i]+n, V_target[0, i]+m, :] = source_img[U_source[1, i], U_source[0, i], :]
    return target_img


# In[26]:


def show_img(file, rotation = 0, size = None):
    #img = mpimg.imread(file)
    img = Image.open(file)
    if size != None:
        img.thumbnail((size, size), Image.ANTIALIAS) # resizes image in-place
    img = ndimage.rotate(img, rotation)
    imgplot = plt.imshow(img)
    plt.show()
    return img

#source_img = show_img('images/guai.jpg', 0, 80)
#fig, ax = plt.subplots(figsize=(15, 15))
target_img = show_img('images/doe.jpg')

#y = source_img.shape[0]
#x = source_img.shape[1]
#U = np.array([[0, x, x, 0], [y, y, 0, 0]])
#V = np.array([[802, 894, 892, 808],[608, 609, 494, 493]])
#V = np.array([[100, 120, 140, 110], [100, 100, 85, 85]])

#target_img = superimpose(U, V, source_img, target_img)
#fig, ax = plt.subplots(figsize=(15, 15))
#plt.imshow(target_img)
#plt.show()


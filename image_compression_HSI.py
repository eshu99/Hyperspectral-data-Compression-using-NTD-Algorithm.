#!/usr/bin/env python
# coding: utf-8

# In[59]:


import spectral


# In[60]:


from spectral import *


# In[61]:


from scipy.io import loadmat


# In[62]:


#loading the hyperspectral dataset
Z = loadmat('PaviaU.mat')
W = loadmat('PaviaU_gt.mat')


# In[63]:


def read_HSI():
  X = loadmat('PaviaU.mat')['paviaU']
  Y = loadmat('PaviaU_gt.mat')['paviaU_gt']
  print(f"X shape: {X.shape}\nY shape: {Y.shape}")
  return X, Y


# In[64]:


X, Y = read_HSI()


# In[65]:


import numpy as  np
import tensorly as tl
import matplotlib.pylab as plt
import time

from tensorly.decomposition import non_negative_tucker
from PIL import Image
from skimage.measure import compare_psnr

random_state = 1234


# In[66]:


data = X
print('Image Shape：' + str(data.shape))


# In[67]:


data_float = data.astype(np.float32)

time0 = time.time()


# In[68]:


#1.compressing the data set using NTD
tucker_ranks = [100, 50, 10]
core, factors = non_negative_tucker(data_float, rank=tucker_ranks, n_iter_max=1000,init='random', tol=0.0001, random_state=random_state, verbose=True)
tucker_tensor=(core, factors)
#storing back the decomposed tensor 
data_reconstruction = tl.tucker_to_tensor(tucker_tensor, transpose_factors=False)


# In[69]:


print('Image Reconstruction Shape：' + str(data_reconstruction.shape))


# In[70]:


im_reconstruction = (data_reconstruction.astype(int))


# In[71]:


mse=[]
comprt=[]
psnr=[]


# In[72]:


#calculating the compression ratio
size_decomposition = sum([factor.size for factor in factors]) + core.size
compression_ratio = (data.size / size_decomposition)
print('Image Compression Ratio：' + str(compression_ratio))
comprt.append(compression_ratio)


# In[73]:


#calculating the psnr
psnr1 = compare_psnr(data, im_reconstruction)
print('Image Compare PSNR：' + str(psnr1))
psnr.append(psnr1)


# In[74]:


#calculating the decomposition time
print('Decomposition Time：' + str(time.time() - time0))


# In[75]:


X.max()


# In[76]:


X.size


# In[77]:


#calculating the mean square error
msqer =  np.mean((im_reconstruction-data)**2)
print('Mean Square Error：' + str(msqer))
mse.append(msqer)


# In[78]:


im_reconstruction.size


# In[79]:


#2.compressing the data set using NTD
tucker_ranks = [200, 40, 10]
core, factors = non_negative_tucker(data_float, rank=tucker_ranks, n_iter_max=1000,init='random', tol=0.0001, random_state=random_state, verbose=True)
tucker_tensor=(core, factors)
data_reconstruction = tl.tucker_to_tensor(tucker_tensor, transpose_factors=False)
im_reconstruction = (data_reconstruction.astype(int))
size_decomposition = sum([factor.size for factor in factors]) + core.size
compression_ratio = (data.size / size_decomposition)
print('Image Compression Ratio：' + str(compression_ratio))
comprt.append(compression_ratio)
psnr1 = compare_psnr(data, im_reconstruction)
print('Image Compare PSNR：' + str(psnr1))
psnr.append(psnr1)
msqer =  np.mean((im_reconstruction-data)**2)
print('Mean Square Error：' + str(msqer))
mse.append(msqer)


# In[80]:


#3.compressing the data set using NTD
tucker_ranks = [50, 25, 5]
core, factors = non_negative_tucker(data_float, rank=tucker_ranks, n_iter_max=1000,init='random', tol=0.0001, random_state=random_state, verbose=True)
tucker_tensor=(core, factors)
data_reconstruction = tl.tucker_to_tensor(tucker_tensor, transpose_factors=False)
im_reconstruction = (data_reconstruction.astype(int))
size_decomposition = sum([factor.size for factor in factors]) + core.size
compression_ratio = (data.size / size_decomposition)
print('Image Compression Ratio：' + str(compression_ratio))
comprt.append(compression_ratio)
psnr1 = compare_psnr(data, im_reconstruction)
print('Image Compare PSNR：' + str(psnr1))
psnr.append(psnr1)
msqer =  np.mean((im_reconstruction-data)**2)
print('Mean Square Error：' + str(msqer))
mse.append(msqer)


# In[81]:


#4.compressing the data set using NTD
tucker_ranks = [300, 150, 30]
core, factors = non_negative_tucker(data_float, rank=tucker_ranks, n_iter_max=1000,init='random', tol=0.0001, random_state=random_state, verbose=True)
tucker_tensor=(core, factors)
data_reconstruction = tl.tucker_to_tensor(tucker_tensor, transpose_factors=False)
im_reconstruction = (data_reconstruction.astype(int))
size_decomposition = sum([factor.size for factor in factors]) + core.size
compression_ratio = (data.size / size_decomposition)
print('Image Compression Ratio：' + str(compression_ratio))
comprt.append(compression_ratio)
psnr1 = compare_psnr(data, im_reconstruction)
print('Image Compare PSNR：' + str(psnr1))
psnr.append(psnr1)
msqer =  np.mean((im_reconstruction-data)**2)
print('Mean Square Error：' + str(msqer))
mse.append(msqer)


# In[82]:


#5.compressing the data set using NTD
tucker_ranks = [100, 40, 20]
core, factors = non_negative_tucker(data_float, rank=tucker_ranks, n_iter_max=1000,init='random', tol=0.0001, random_state=random_state, verbose=True)
tucker_tensor=(core, factors)
data_reconstruction = tl.tucker_to_tensor(tucker_tensor, transpose_factors=False)
im_reconstruction = (data_reconstruction.astype(int))
size_decomposition = sum([factor.size for factor in factors]) + core.size
compression_ratio = (data.size / size_decomposition)
print('Image Compression Ratio：' + str(compression_ratio))
comprt.append(compression_ratio)
psnr1 = compare_psnr(data, im_reconstruction)
print('Image Compare PSNR：' + str(psnr1))
psnr.append(psnr1)
msqer =  np.mean((im_reconstruction-data)**2)
print('Mean Square Error：' + str(msqer))
mse.append(msqer)


# In[83]:


psnr


# In[84]:


mse


# In[85]:


comprt


# In[86]:


#Graph showing variation of Mean Squared Error with the Compression Ratio
import matplotlib.pyplot as plt
x1 = comprt
y1 = mse
plt.plot(x1, y1, label = "MSE")
plt.xlabel("Compression Ratio")
plt.ylabel("MSE")
plt.legend()
plt.show()


# In[87]:


#Graph showing variation of PSNR with the Compression Ratio
x2 = comprt
y2 = psnr
plt.plot(x2, y2, label = "PSNR")
plt.xlabel("Compression Ratio")
plt.ylabel("PSNR")
plt.legend()
plt.show()


# In[ ]:





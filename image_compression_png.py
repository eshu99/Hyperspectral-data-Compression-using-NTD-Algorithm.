#!/usr/bin/env python
# coding: utf-8

# In[5]:


# importing matplotlib module 
from matplotlib import pyplot as plt 

# x-axis values 
x = [5, 2, 9, 4, 7] 

# Y-axis values 
y = [10, 5, 8, 4, 2] 

# Function to plot 
plt.plot(x, y) 

# function to show the plot 
plt.show() 


# In[6]:


import numpy as  np
import tensorly as tl


# In[7]:


from tensorly.decomposition import non_negative_tucker


# In[8]:


random_state = 1234


# In[9]:


import matplotlib.pylab as plt


# In[10]:


from PIL import Image


# In[11]:


from skimage.measure import compare_psnr


# In[15]:


im = Image.open("original.png")


# In[16]:


print('Image Mode and Size：' + str(im.mode) + ',' + str(im.size))


# In[17]:


data = np.array(im)
print('Image Shape：' + str(data.shape))


# In[18]:


import time


# In[19]:


data_float = data.astype(np.float32)

time0 = time.time()


# In[22]:


tucker_ranks = [50, 50, 3]
core, factors = non_negative_tucker(data_float, rank=tucker_ranks, n_iter_max=1000,init='random', tol=0.0001, random_state=random_state, verbose=True)
tucker_tensor=(core, factors)
data_reconstruction = tl.tucker_to_tensor(tucker_tensor, skip_factor=None, transpose_factors=False)


# In[23]:


print('Image Reconstruction Shape：' + str(data_reconstruction.shape))


# In[24]:


def convert2uin8(tensor):
    im = tl.to_numpy(tensor)
    im -= im.min()
    im /= im.max()
    im *= 255
    return im.astype(np.uint8)


im_reconstruction = convert2uin8(data_reconstruction)


# In[25]:


fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.set_axis_off()
ax.imshow(im)
ax.set_title('Original')

ax = fig.add_subplot(1, 2, 2)
ax.set_axis_off()
ax.imshow(im_reconstruction)
ax.set_title('Tucker')

plt.tight_layout()
plt.savefig(str(tucker_ranks[0]) + '.jpg')


# In[26]:


plt.show()


# In[27]:


size_decomposition = sum([factor.size for factor in factors]) + core.size
compression_ratio = (data.size / size_decomposition)
print('Image Compression Ratio：' + str(compression_ratio))


# In[28]:


psnr = compare_psnr(data, im_reconstruction)
print('Image Compare PSNR：' + str(psnr))


# In[29]:


print('Decomposition Time：' + str(time.time() - time0))



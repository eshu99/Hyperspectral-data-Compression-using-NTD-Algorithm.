#!/usr/bin/env python
# coding: utf-8

# In[71]:

import numpy as np
import tensorly as tl

# In[72]:

tensor = tl.tensor(np.arange(48).reshape((6, 4, 2)), dtype=tl.float32)

# In[73]:

tensor.shape

# In[74]:

tensor

# In[75]:

tensor[:,:,0]

# In[76]:

tensor[:,:,1]

# In[77]:

type(tensor)

# In[78]:

tl.unfold(tensor,mode=0)

# In[79]:

tl.unfold(tensor,mode=1)

# In[80]:

tl.unfold(tensor,mode=2)

# In[81]:

unfolding = tl.unfold(tensor, 1)
original_shape = tensor.shape
tl.fold(unfolding, mode=1, shape=original_shape)

# In[82]:

from tensorly.decomposition import non_negative_tucker
random_state = 1234

# In[96]:

core, factors = non_negative_tucker(tensor, rank=[3,2,1], n_iter_max=10, init='random', tol=0.0001, random_state=None, verbose=False, ranks=None)

# In[97]:

core.shape

# In[98]:

core

# In[99]:

len(factors)

# In[100]:

[f.shape for f in factors]


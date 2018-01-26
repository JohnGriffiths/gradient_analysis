
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import scipy as sp
from scipy import io
from sklearn.metrics import pairwise_distances

import sys
#sys.path.append("../topography/utils_py/mapalign")
sys.path.append('/home/hpc3230/Code/libraries_of_others/github/mapalign')
from mapalign import dist


# In[14]:


atlases = {'rm': 0, 'lv': 1, 'fv': 3, 'b05': 4, 'bb47': 5, 'pht00': 7}

def create_macaque_conn_mat(mat_name):

    col = atlases.get(mat_name)
    mat = pd.read_csv(('gradient_data/macaque/%s_mat.txt' % mat_name), delimiter='\t', header=-1)
    nam = pd.read_csv(('gradient_data/macaque/%s_name.txt' % mat_name), delimiter='\n', header=-1)[0]
    m = np.array(mat)[:,:-1]
    
    toRemove = np.where(m.sum(axis=0) != 0)[0]
    m = m[toRemove,:][:,toRemove].copy()
    nam = nam[toRemove].copy()
    
    # transpose and concatenate to include bidirectional connectivity:
    m = np.hstack((m,m.T))
        
    # Calculate pairwise Euclidean distance:
    aff = dist.compute_affinity(m)
    
    return aff, nam

def save_macaque_matrices(mat_name):
    
    conn_mat, names = create_macaque_conn_mat(mat_name)
    
    np.save('gradient_data/conn_matrices/macaque_%s_conn.npy' % mat_name, conn_mat)
    np.save('gradient_data/conn_matrices/macaque_%s_names.npy' % mat_name, names)


# In[15]:


save_macaque_matrices('bb47')


# In[ ]:





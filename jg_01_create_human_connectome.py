
# coding: utf-8

# In[1]:


# Download HCP 820 dense connectome:
# HCP_S900_820_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.dconn.nii


# In[ ]:


import numpy as np
import nibabel as nib # git clone --branch enh/cifti2 https://github.com/satra/nibabel.git
from sklearn.metrics import pairwise_distances

# Load data and Fisher's z-to-r transform
#
f = '/scratch/hpc3230/Data/connectomeDB/HCP_S900_820_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.dconn.nii'

#dcon = np.tanh(nib.load('HCP_S900_820_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.dconn.nii').data)

dcon = np.tanh(np.squeeze(nib.load(f).get_data()))

# Get number of nodes
N = dcon.shape[0]

# Generate percentile thresholds for 90th percentile
perc = np.array([np.percentile(x, 90) for x in dcon])

# Threshold each row of the matrix by setting values below 90th percentile to 0
for i in range(dcon.shape[0]):
  print "Row %d" % i
  dcon[i, dcon[i,:] < perc[i]] = 0    


# In[ ]:


# Check for minimum value
print "Minimum value is %f" % dcon.min()

# The negative values are very small, but we need to know how many nodes have negative values
# Count negative values per row
neg_values = np.array([sum(dcon[i,:] < 0) for i in range(N)])
print "Negative values occur in %d rows" % sum(neg_values > 0)

# Since there are only 23 vertices with total of 5000 very small negative values, we set these to zero
dcon[dcon < 0] = 0


# In[ ]:

# JG_ADD: eigenvector decomposition of FC matrix
print 'computing eigenvectors of 90th percentile FC matrix'
dcon_evals,dcon_evecs = np.linalg.eig(dcon)
np.save('gradient_data/conn_matrices/dcon_evals.npy', dcon_evals)
np.save('gradient_data/conn_matrices/dcon_evecs.npy', dcon_evecs)



# Now we are dealing with sparse vectors. Cosine similarity is used as affinity metric
aff = 1 - pairwise_distances(dcon, metric = 'cosine')


# In[ ]:


# Save affinity matrix
np.save('gradient_data/conn_matrices/cosine_affinity.npy', aff)




# JG_ADD: eigenvector decomposition of affinity matrix
print 'computing eigenvectors of affinity matrix'
aff_evals,aff_evecs = np.linalg.eig(aff)
np.save('gradient_data/conn_matrices/aff_evals.npy', aff_evals)
np.save('gradient_data/conn_matrices/aff_evecs.npy', aff_evecs)









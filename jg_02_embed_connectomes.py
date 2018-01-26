
# coding: utf-8

# # Human connectivity embedding

# In[2]:


import numpy as np


# Load affinitity matrix
aff = np.load('gradient_data/conn_matrices/cosine_affinity.npy')


# In[ ]:


import sys
#sys.path.append("../topography/utils_py/mapalign")
sys.path.append('/home/hpc3230/Code/libraries_of_others/github/mapalign')
from mapalign import embed

#emb, res = embed.compute_diffusion_map(aff, alpha = 0.5)
emb, res = embed.compute_diffusion_map(aff, alpha = 0.5,return_result=True) # JG_MOD


# In[ ]:


# Save results
np.save('gradient_data/embedded/embedding_dense_emb.npy', emb)
np.save('gradient_data/embedded/embedding_dense_res.npy', res)


# In[1]:


import numpy as np
res = np.load('gradient_data/embedded/embedding_dense_res.npy').item()
a = [res['vectors'][:,i]/ res['vectors'][:,0] for i in range(302)]
emb = np.array(a)[1:,:].T
len(emb)


# ## Export to cifti space

# In[2]:


import nibabel as nib
import numpy as np


# In[3]:

# JG_MOD
#res = nib.load('gradient_data/templates/hcp.tmp.lh.dscalar.nii').data 
img = nib.load('gradient_data/templates/hcp.tmp.lh.dscalar.nii')
res = np.squeeze(img.get_data())
cortL = np.squeeze(np.array(np.where(res != 0)[0], dtype=np.int32))

#res = nib.load('gradient_data/templates/hcp.tmp.rh.dscalar.nii').data
img = nib.load('gradient_data/templates/hcp.tmp.rh.dscalar.nii')
res = np.squeeze(img.get_data())
cortR = np.squeeze(np.array(np.where(res != 0)[0], dtype=np.int32))

cortLen = len(cortL) + len(cortR)
del res

# save out cortR and cortL in gradient_data/templates/


# In[4]:


emb = np.load('gradient_data/embedded/embedding_dense_emb.npy')


# In[5]:

# JG_MOD
#tmp = nib.nifti2.load('gradient_data/templates/100307_tfMRI_MOTOR_level2_hp200_s2.dscalar.nii')
#tmp_cifti = nib.cifti.load('gradient_data/templates/100307_tfMRI_MOTOR_level2_hp200_s2.dscalar.nii')
tmp = nib.load('gradient_data/templates/100307_tfMRI_MOTOR_level2_hp200_s2.dscalar.nii')
#data = tmp_cifti.data * 0

data = tmp.dataobj[:] * 0
#data[0:10,:len(emb)] = np.reshape(emb.T, [1, 1, 1, 1] + list(emb.T.shape))
data[0:9,:len(emb)] = np.reshape(emb.T, [1, 1, 1, 1] + list(emb.T.shape))

#tmp2 = nib.nifti2.Nifti2Image(data, tmp.get_affine())
#tmp4 = nib.nifti2.create_cifti_image(tmp2, tmp_cifti.header.to_xml(), np.array(3006, dtype=np.int32))

tmp4 = nib.Cifti2Image(data,tmp.header)
tmp4.to_filename('gradient_data/embedded/ciftis/hcp.embed.dscalar.nii')


# # Macaque connectivity embedding

# In[6]:


import sys
#sys.path.append("../topography/utils_py/mapalign")
from mapalign import embed
    
def embed_macaque(mat_name):
    
    aff = np.load('gradient_data/conn_matrices/macaque_%s_conn.npy' % mat_name)
    print np.shape(aff)
    #emb, res = embed.compute_diffusion_map(aff, alpha = 0.5)
    emb, res = embed.compute_diffusion_map(aff, alpha = 0.5, return_result=True) # JG_MOD
    np.save('gradient_data/conn_matrices/macaque_%s_emb.npy' % mat_name, emb)
    np.save('gradient_data/conn_matrices/macaque_%s_res.npy' % mat_name, res)


# In[7]:


embed_macaque('bb47')


# In[ ]:





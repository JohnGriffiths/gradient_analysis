
# coding: utf-8

# # Components variance

# In[1]:


get_ipython().magic(u'matplotlib inline')

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['svg.fonttype'] = 'none'

emb = np.load('gradient_data/embedded/embedding_dense_res.npy')


# In[2]:


lam = np.squeeze(np.array(list(emb.item()['orig_lambdas'].flatten())))[1::]

fig, (ax) = plt.subplots(1,2, figsize=(15,7), facecolor='white')

vals = lam #/ np.sum(lam)
vals = np.hstack((0,vals))
ax[0].semilogx(vals, marker='o', mfc='k', mec='k', linestyle='-', color='k')
ax[0].set_ylim([-0.025,np.max(vals)+0.025])
ax[0].set_xlim(xmin=.8)
ax[0].set_title('Component Lambdas')

vals = np.cumsum(lam) #/ np.sum(lam)
vals = np.hstack((0,vals))
ax[1].semilogx(vals, marker='o', mfc='k', mec='k', linestyle='-', color='k')
ax[1].set_ylim([np.min(vals) - 0.025,np.max(vals)+0.025])
ax[1].set_xlim(xmin=.8)
ax[1].set_title('Component Cumulative Lambdas')

ax[0].set_xlabel('log(Component)')
ax[0].set_ylabel('Lambda')
ax[1].set_xlabel('log(Component)')
ax[1].set_ylabel('Cumulative Lambdas')

fig.savefig('gradient_data/figures/Fig.supp.human_componentVariance.lambdas.svg', format='svg')
fig.savefig('gradient_data/figures/Fig.supp.human_componentVariance.lambdas.png')
plt.show()


# In[28]:


lam = np.squeeze(np.array(list(emb.item()['lambdas'].flatten())))

fig, (ax) = plt.subplots(1,2, figsize=(15,7), facecolor='white')

vals = lam / np.sum(lam)
vals = np.hstack((0,vals))
ax[0].semilogx(vals, marker='o', mfc='k', mec='k', linestyle='-', color='k')
ax[0].set_ylim([-0.025,np.max(vals)+0.025])
ax[0].set_xlim(xmin=.8)
ax[0].set_title('Component variance')

vals = np.cumsum(lam) / np.sum(lam)
vals = np.hstack((0,vals))
ax[1].semilogx(vals, marker='o', mfc='k', mec='k', linestyle='-', color='k')
ax[1].set_ylim([np.min(vals) - 0.025,np.max(vals)+0.025])
ax[1].set_xlim(xmin=.8)
ax[1].set_title('Component cumulative variance')

ax[0].set_xlabel('log(Component)')
ax[0].set_ylabel('Variance')
ax[1].set_xlabel('log(Component)')
ax[1].set_ylabel('Cumulative variance')

fig.savefig('gradient_data/figures/Fig.supp.human_componentVariance.lambdas.svg', format='svg')
fig.savefig('gradient_data/figures/Fig.supp.human_componentVariance.lambdas.png')
plt.show()


# In[29]:


np.shape(lam)


# In[ ]:





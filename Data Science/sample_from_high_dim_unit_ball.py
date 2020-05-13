#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# In[2]:


class r_pdf(stats.rv_continuous):
    def _pdf(self, x, dim):
        return dim*(x**(dim-1))

def uniform_sample_from_d_dim_unit_ball(dim, num_points):
    # samples from multivariate normal distribution where mean = 0 and covariance 1
    mean = np.zeros(dim)
    cov = np.eye(dim)
    g_samples = np.random.multivariate_normal(mean, cov, num_points) # samples from multivariate gaussian
    
    return g_samples
    
def plot_samples_2D(samples):
    
    plt.scatter(samples[:,0], samples[:,1])
    plt.show()
    
    


# In[3]:


dim = 2
num_samples = 1000
g_samples = uniform_sample_from_d_dim_unit_ball(dim,num_samples) # sample points on the surface of unit ball
norm = np.reshape(np.linalg.norm(g_samples, axis=1), (-1,1)) # compute the norm for each sample
g_samples = g_samples/norm #normalized gaussian samples

r = r_pdf(a = 0, b = 1, name = 'radius')
r_samples = r.rvs(dim=dim,size = num_samples) # sample the radius to compute points inside the unit ball
r_samples = np.reshape(r_samples, (-1,1))

print(np.shape(g_samples), np.shape(r_samples*g_samples))


# In[4]:


samples = r_samples*g_samples
print(samples)



plot_samples_2D(samples)


# In[ ]:





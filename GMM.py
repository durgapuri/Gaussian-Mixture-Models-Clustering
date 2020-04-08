#!/usr/bin/env python
# coding: utf-8

# In[99]:


import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from scipy.stats import norm
import pickle
np.random.seed(0)
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


# ### Load File

# In[100]:


def load(name):
    file = open(name,'rb')
    data = pickle.load(file)
    file.close()
    return data


# ### Save File

# In[101]:


def save(data,name):
    file = open(name, 'wb')
    pickle.dump(data,file)
    file.close()


# ### GMM 

# In[102]:


class GMM1D:
    def __init__(self,X,iterations,initmean,initprob,initvariance):  
        self.iterations = iterations
        self.X = X
        self.mu = initmean
        self.pi = initprob
        self.var = initvariance
    
    """E step"""

    def calculate_prob(self,r):
        for c,g,p in zip(range(3),[norm(loc=self.mu[0],scale=self.var[0]),
                                       norm(loc=self.mu[1],scale=self.var[1]),
                                       norm(loc=self.mu[2],scale=self.var[2])],self.pi):
            self.X = self.X.flatten()
            val = 2 + c
            r[:,c] = p*g.pdf(self.X)
        """
        Normalize the probabilities such that each row of r sums to 1 and weight it by mu_c == the fraction of points belonging to 
        cluster c
        """

        for i in range(len(r)):
        	# Write code here
            n_frac = r[i]
            d_frac = np.sum(self.pi)*np.sum(r, axis=1)[i]
            r[i] = n_frac/d_frac
            pass
        return r
    
    def plot(self,r):
        fig = plt.figure(figsize=(10,10))
        ax0 = fig.add_subplot(111)
        for i in range(len(r)):
            ax0.scatter(self.X[i],0,c=np.array([r[i][0],r[i][1],r[i][2]]),s=100)
        """Plot the gaussians"""
        for g,c in zip([norm(loc=self.mu[0],scale=self.var[0]).pdf(np.linspace(-20,20,num=60)),
                        norm(loc=self.mu[1],scale=self.var[1]).pdf(np.linspace(-20,20,num=60)),
                        norm(loc=self.mu[2],scale=self.var[2]).pdf(np.linspace(-20,20,num=60))],['r','g','b']):
            ax0.plot(np.linspace(-20,20,num=60),g,c=c)
    
    def run(self):
        
        for iter in range(self.iterations):

            """Create the array r with dimensionality nxK"""
            r = np.zeros((len(self.X),3)) 

            """
            Probability for each datapoint x_i to belong to gaussian g 
            """
            r = self.calculate_prob(r)

            """Plot the data"""
            self.plot(r)
            
            """M-Step"""
            
            rows, cols = r.shape
            """calculate m_c"""
            m_c = []
            # write code here
            for c in range(cols):
                sum_cl = np.sum(r[:,c])
                d=c
                m_c.append(sum_cl)
                
            
            """calculate pi_c"""
            # write code here
            for c in range(cols):
                n_frac = m_c[c]
                d = c
                d_frac = np.sum(m_c)
                d = 0
                self.pi[c] = (n_frac/d_frac)
                
            
            """calculate mu_c"""
            # write code here
            d_frac = m_c
            self.mu = np.sum(self.X.reshape(rows,1)*r,axis=0)/d_frac
            

            """calculate var_c"""
            var_c = []
            #write code here
            for c in range(cols):
                var1 = (1/m_c[c])
                d=c
                var_c.append(var1 *np.dot(((np.array(r[:,c]).reshape(rows,1))*(self.X.reshape(rows,1)-self.mu[c])).T,(self.X.reshape(rows,1)-self.mu[c])))
                d=0
                
            plt.show()

# """
# To run the code - 
# g = GMM1D(data,10,[mean1,mean2,mean3],[1/3,1/3,1/3],[var1,var2,var3])
# g.run()
# """


# ### Loading and Merging Data

# In[103]:


data = load("/home/jyoti/Documents/SMAI/assign2/Assignment-2_Dataset/Datasets/Question-2/dataset1.pkl")
data1 = load("/home/jyoti/Documents/SMAI/assign2/Assignment-2_Dataset/Datasets/Question-2/dataset2.pkl")
data2 = load("/home/jyoti/Documents/SMAI/assign2/Assignment-2_Dataset/Datasets/Question-2/dataset3.pkl")
final_data = np.append(data, data1)
final_data = np.append(final_data, data2)
g = GMM1D(final_data,10,[-8,8,5],[1/3,1/3,1/3],[5,3,1])
g.run()


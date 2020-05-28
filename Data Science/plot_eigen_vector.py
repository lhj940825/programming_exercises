'''
 * User: Hojun Lim
 * Date: 2020-05-23
'''

import numpy as np
import matplotlib.pyplot as plt
A = np.array([[2,0,4], [0, -1, 0], [4, 0 ,3]])

t = np.linspace(0,np.pi*2,50)
v = np.asarray(list(zip(np.cos(t), np.zeros(len(t)), np.sin(t))))
Av = np.matmul(A,v.T).T

plt.plot(np.cos(t), np.sin(t), linewidth=2, label='unit circle')
plt.plot(Av[:,0], Av[:,2], linewidth=2, label = 'unit circle transformed by the matrix A')

eigen_v1 = np.array([(np.sqrt(65)-1)/8, 0, 1])
eigen_v2 = np.array([(-np.sqrt(65)-1)/8,0, 1])

A_eigen_v1= np.matmul(A, eigen_v1).T
A_eigen_v2= np.matmul(A, eigen_v2).T

eigen_v = np.array([eigen_v1, eigen_v2])
A_eigen_v = np.array([A_eigen_v1, A_eigen_v2])

origin = [0], [0] # origin point
plt.arrow(0,0, eigen_v1[0], eigen_v1[2],head_width=0.3, head_length=0.3, color = 'r', label='eigen vector')
plt.arrow(0,0, eigen_v2[0], eigen_v2[2], head_width=0.3, head_length=0.3, color = 'r', label='eigen vector')
plt.arrow(0,0, A_eigen_v1[0], A_eigen_v1[2],head_width=0.3, head_length=0.3, color = 'g', label="eigen value*eigen vector")
plt.arrow(0,0, A_eigen_v2[0], A_eigen_v2[2],head_width=0.3, head_length=0.3, color = 'g', label="eigen value*eigen vector")

plt.xlim(-8,8)
plt.ylim(-8,8)
plt.xlabel('x axis')
plt.ylabel('z axis')
plt.legend(loc="upper left")
plt.show()
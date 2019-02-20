# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:52:59 2019

@author: Yu Chen
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
from pylab import *
import numpy as np

'''
================ Part 1: Feature Normalization ================
'''
print("Loading Data \n")
data = np.loadtxt('ex1data2.txt', delimiter=",")
m, n = data.shape  # m is number of samples and n is number of features

x = data[:,[0,1]].reshape((m,2))
y = data[:,2].reshape((m,1))

print("Normalizing Features \n")

def featureNormalize(Xi):
    m, n = Xi.shape
    X_norm = np.zeros((m,n))
    for i in range(n):
        mean = np.mean(Xi[:,i])
        std = np.std(Xi[:,i])
        X_norm[:,i] = (Xi[:,i] - mean) / std
    return X_norm

X = featureNormalize(x)
# add one column of 1 to the X
X = np.append(np.ones((m,1)), X, axis=1) #np.concatenate(np.ones((m,1)), x, axis=1)  
'''
=================== Part 2: Gradient descent ===================
'''
print("Running Gradient Descent \n")
theta = np.zeros((3,1))

# Some gradient descent settings
iterations = 400;
alpha = 0.01;

def computerCostMulti(xi, yi, thetai):
    m = yi.size
    J = 0
    for i in range(m):
        dhx = np.dot(xi, thetai) - yi
        J = 1/2/m*np.dot(np.transpose(dhx), dhx)
    return J

def gradientDescentMulti(xi, yi, thetai, alphai, iteri):
    m = yi.size
    J_history = np.zeros((iteri, 1))
    for i in range(iteri):
        dhx = np.transpose(np.dot(xi, thetai) - yi)
        thetai = thetai - alphai/m*np.transpose(np.dot(dhx, xi))
        J_history[i] = computerCostMulti(xi, yi, thetai)
    return thetai, J_history

theta_final, J_hist = gradientDescentMulti(X, y, theta, alpha, iterations)

print("Theta found by gradient descent are %f, %f and %f.\n"%(theta_final[0], theta_final[1], theta_final[2]))

Xp = np.linspace(1,J_hist.size, J_hist.size)
plt.plot(Xp, J_hist, 'r-')
plt.xlabel("# of Iterations")
plt.ylabel("Cost J")
plt.legend(loc='upper right')
plt.show()


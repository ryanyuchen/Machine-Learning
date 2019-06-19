# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:16:08 2019

@author: Yu Chen
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
from pylab import *
import numpy as np

'''
======================= Part 2: Plotting =======================
'''
print("Plotting Data \n")
data = np.loadtxt('ex1data1.txt', delimiter=",")
m, n = data.shape  # m is number of samples and n is number of features

x = data[:,0].reshape((m,1))
y = data[:,1].reshape((m,1))
plt.scatter(x,y, label="Training Data")
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.legend(loc='upper right')
plt.show()
'''
=================== Part 3: Gradient descent ===================
'''
print("Running Gradient Descent \n")
# add one column of 1 to the x
X = np.append(np.ones((m,1)), x, axis=1) #np.concatenate(np.ones((m,1)), x, axis=1)
theta = np.zeros((2,1))

# Some gradient descent settings
iterations = 1500;
alpha = 0.01;

def computerCost(xi, yi, thetai):
    m = yi.size
    J = 0
    for i in range(m):
        J = J + 1/2/m*np.square(np.dot(xi[i,:], thetai) - yi[i])
    return J

print(computerCost(X, y, theta))

def gradientDescent(xi, yi, thetai, alphai, iteri):
    m = yi.size
    J_history = np.zeros((iteri, 1))
    for i in range(iteri):
        hx = np.transpose(np.dot(xi, thetai) - yi)
        thetai = thetai - alphai/m*np.transpose(np.dot(hx, xi))
        J_history[i] = computerCost(xi, yi, thetai)
    return thetai, J_history

theta_final, J_hist = gradientDescent(X, y, theta, alpha, iterations)

print("Theta found by gradient descent are %f and %f.\n"%(theta_final[0], theta_final[1]))

plt.scatter(x,y, label="Training Data")
plt.plot(X[:,1], np.dot(X, theta_final), 'r-', label="Linear Regression")
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.legend(loc='upper right')
plt.show()

predict1 = np.dot(np.array([1, 3.5]), theta_final)
print("For population = 35,000, we predict a profit of %f\n"%(predict1*10000))

'''
============= Part 4: Visualizing J(theta_0, theta_1) =============
'''
print("Visualizing J(theta_0, theta_1) \n")
theta0_vals = np.linspace(-10,10,100)
theta1_vals = np.linspace(-1,4,100)

J_vals = np.zeros((theta0_vals.size, theta1_vals.size))

for i in range(theta0_vals.size):
    for j in range(theta1_vals.size):
        tmp = np.array([theta0_vals[i], theta1_vals[j]]).reshape((2,1))
        J_vals[i,j] = computerCost(X, y, tmp)

J_vals = np.transpose(J_vals)

fig = plt.figure()
ax = fig.gca(projection='3d')

Xp, Yp = np.meshgrid(theta0_vals, theta1_vals)
surf = ax.plot_surface(Xp, Yp, J_vals, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()

fig = plt.figure()
cp = plt.contour(Xp, Yp, J_vals)
plt.show()

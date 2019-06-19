# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:54:50 2019

@author: Yu Chen
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin_bfgs

'''
Load Data and Plotting
'''
print("Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples \n")
# Loading Data
data = np.loadtxt('ex2data2.txt', delimiter=",")
m, n = data.shape  # m is number of samples and n is number of features

#Plotting Data
x = data[:,[0,1]].reshape((m,2))
y = data[:,2].reshape((m,1))
pos = np.where(y == 1)[0]
neg = np.where(y == 0)[0]
plt.plot(x[pos,0],x[pos,1], 'b+', linewidth=4, markersize=7, label="y=1")
plt.plot(x[neg,0],x[neg,1], 'ro', markersize=7, label="y=0")
plt.xlabel("Microchip Test 1")
plt.ylabel("Microchip Test 2")
plt.legend(loc='upper right')
plt.show()

'''
=========== Part 1: Regularized Logistic Regression ============
'''
def mapFeature(X1, X2):
    degree = 6
    out = np.ones(( X1.shape[0], sum(range(degree + 2)) )) # could also use ((degree+1) * (degree+2)) / 2 instead of sum
    curr_column = 1
    for i in range(1, degree + 1):
        for j in range(i+1):
            out[:,curr_column] = np.power(X1,i-j) * np.power(X2,j)
            curr_column += 1

    return out

def sigmoid(zi):
    #g = np.zeros(zi.shape[0])
    g = 1/(1 + np.exp(-zi))
    return g

def costFunctionReg(thetai, xi, yi, lambda_regi, return_grad=False):
    m = len(yi) 
    J = 0
    grad = np.zeros(thetai.shape)

    one = yi * np.transpose(np.log( sigmoid( np.dot(xi,thetai) ) ))
    two = (1-yi) * np.transpose(np.log( 1 - sigmoid( np.dot(xi,thetai) ) ))
    reg = ( float(lambda_regi) / (2*m)) * np.power(thetai[1:thetai.shape[0]],2).sum()
    J = -(1./m)*(one+two).sum() + reg

    # applies to j = 1,2,...,n - NOT to j = 0
    grad = (1./m) * np.dot(sigmoid( np.dot(xi,thetai) ).T - yi, xi).T + ( float(lambda_regi) / m )*thetai

    # the case of j = 0 (recall that grad is a n+1 vector)
    # since we already have the whole vectorized version, we use that
    grad_no_regularization = (1./m) * np.dot(sigmoid( np.dot(xi,thetai) ).T - yi, xi).T

    # and then assign only the first element of grad_no_regularization to grad
    grad[0] = grad_no_regularization[0]

    if return_grad == True:
        return J, grad.flatten()
    elif return_grad == False:
        return J      

X = mapFeature(X[:,0], X[:,1])
m,n = X.shape

# Initialize fitting parameters
initial_theta = np.zeros((n, 1))

# Set regularization parameter lambda to 1
lambda_reg = 0.1

cost = costFunctionReg(initial_theta, X, y, lambda_reg)
print("Cost at initial theta (zeros):  %f.\n"%(cost))

'''
============= Part 2: Regularization and Accuracies =============
'''
# Initialize fitting parameters
initial_theta = np.zeros((n, 1))

# Set regularization parameter lambda to 1 (you should vary this)
lambda_reg = 1

#  Run fmin_bfgs to obtain the optimal theta
#  This function returns theta and the cost 
myargs=(X, y, lambda_reg)
theta = fmin_bfgs(costFunctionReg, x0=initial_theta, args=myargs)

def plotDecisionBoundary(thetai, xi, yi):
    pos = np.where(yi == 1)[0]
    neg = np.where(yi == 0)[0]
    plt.plot(xi[pos,1],xi[pos,2], 'b+', linewidth=4, markersize=7, label="Admitted")
    plt.plot(xi[neg,1],xi[neg,2], 'ro', markersize=7, label="Not Admitted")
    plt.xlabel("Exam 1 Score")
    plt.ylabel("Exam 2 Score")
    plt.legend(loc='upper right')
    
    if (xi.shape[1] <= 3):
        plot_x = np.array([np.min(xi[:,1])-2, np.max(xi[:,1])+2])
        plot_y = (-1 / thetai[2]) * (thetai[1] * plot_x + thetai[0])
        plt.plot(plot_x, plot_y, linewidth=4, label="Decision Boundary")
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros(( len(u), len(v) ))
        
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = np.dot(mapFeature(np.array([u[i]]), np.array([v[j]])),thetai)
        z = np.transpose(z)
        p3 = plt.contour(u, v, z, levels=[0], linewidth=2).collections[0]
        # Legend, specific for the exercise
        #plt.legend((p1,p2, p3),('y = 1', 'y = 0', 'Decision Boundary'), numpoints=1, handlelength=0)
    
plotDecisionBoundary(theta, X, y)

def predict(thetai, xi):
    m = xi.shape[0] 
    p = np.zeros((m, 1))
    sigValue = sigmoid( np.dot(xi,thetai.T) )
    p = sigValue >= 0.5

    return p

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: {:f}'.format(np.mean(p == y) * 100))
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:37:15 2019

@author: Yu Chen
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op

'''
==================== Part 1: Plotting ====================
'''
print("Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples \n")
# Loading Data
data = np.loadtxt('ex2data1.txt', delimiter=",")
m, n = data.shape  # m is number of samples and n is number of features

#Plotting Data
x = data[:,[0,1]].reshape((m,2))
y = data[:,2].reshape((m,1))
pos = np.where(y == 1)[0]
neg = np.where(y == 0)[0]
plt.plot(x[pos,0],x[pos,1], 'b+', linewidth=4, markersize=7, label="Admitted")
plt.plot(x[neg,0],x[neg,1], 'ro', markersize=7, label="Not Admitted")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend(loc='upper right')
plt.show()

'''
============ Part 2: Compute Cost and Gradient ============
'''
mx, nx = x.shape
X = np.append(np.ones((mx,1)), x, axis=1) #np.concatenate(np.ones((m,1)), x, axis=1)
initial_theta = np.zeros(nx+1)

def sigmoid(zi):
    #g = np.zeros(zi.shape[0])
    g = 1/(1 + np.exp(-zi))
    return g
     
def costFunction(xi, yi, thetai):
    m = yi.size
    J = 0
    grad = np.zeros(thetai.shape)
    
    xi = np.matrix(xi)
    yi = np.matrix(yi)
    thetai = np.matrix(thetai)
    
    z = xi * thetai.T
    ht = sigmoid(z)
    J = np.array(1/m * (np.dot(np.transpose(-yi), np.log(ht)) - np.dot(np.transpose(1-yi), np.log(1-ht))))
    grad = np.array(1/m * np.dot(np.transpose(ht-y), X)).reshape(3,1)
    return J[0], grad

cost, grad = costFunction(X, y, initial_theta)
print("Cost at initial theta (zeros):  %f.\n"%(cost[0]))
print("Gradient at initial theta (zeros): %f, %f and %f.\n"%(grad[0], grad[1], grad[2]))

'''
============= Part 3: Optimizing using fminunc  =============
'''
def CostFunc(thetai, xi, yi):
    m = yi.size
    J = 0
    
    xi = np.matrix(xi)
    yi = np.matrix(yi)
    thetai = np.matrix(thetai)
    
    z = xi * thetai.T
    ht = sigmoid(z)
    J = np.array(1/m * (np.dot(np.transpose(-yi), np.log(ht)) - np.dot(np.transpose(1-yi), np.log(1-ht))))
    return J[0]

def GradientFunc(thetai, xi, yi):
    m = yi.size
    grad = np.zeros(thetai.shape)
    
    xi = np.matrix(xi)
    yi = np.matrix(yi)
    thetai = np.matrix(thetai)
    
    z = xi * thetai.T
    ht = sigmoid(z)
    grad = np.array(1/m * np.dot(np.transpose(ht-y), X)).reshape(3,1)
    return grad    

Result = op.minimize(fun=CostFunc, x0=initial_theta, args=(X,y), method='TNC', jac=GradientFunc)
print("Cost at theta found by TNC method:  %f.\n"%(Result.fun[0]))
print("Theta found by TNC method: %f, %f and %f.\n"%(Result.x[0], Result.x[1], Result.x[2]))

def mapFeature(X1, X2):
    degree = 6
    out = np.ones(( X1.shape[0], sum(range(degree + 2)) )) # could also use ((degree+1) * (degree+2)) / 2 instead of sum
    curr_column = 1
    for i in xrange(1, degree + 1):
        for j in xrange(i+1):
            out[:,curr_column] = np.power(X1,i-j) * np.power(X2,j)
            curr_column += 1

    return out

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
        plt.legend((p1,p2, p3),('y = 1', 'y = 0', 'Decision Boundary'), numpoints=1, handlelength=0)
    
resultedTheta = Result.x
plotDecisionBoundary(resultedTheta, X, y)

'''
============== Part 4: Predict and Accuracies ==============
'''
def predict(thetai, xi):
    m = xi.shape[0] 
    p = np.zeros((m, 1))
    sigValue = sigmoid( np.dot(xi,thetai.T) )
    p = sigValue >= 0.5

    return p

prob = sigmoid(np.dot(np.array([1,45,85]),resultedTheta))
print('For a student with scores 45 and 85, we predict an admission probability of {:f}'.format(prob))

# Compute accuracy on our training set
p = predict(resultedTheta, X)

print('Train Accuracy: {:f}'.format(np.mean(p == y) * 100))
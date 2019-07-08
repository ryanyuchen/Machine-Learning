# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:37:10 2019

@author: Yu Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import scipy.io

'''
============ Part 1: Loading and Visualizing Data ============
'''
print("Loading and Visualizing Data ...\n")
def plotData(x, y):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    plt.plot(x[pos,0],x[pos,1], 'k+', linewidth=1, markersize=7)
    plt.plot(x[neg,0],x[neg,1], 'ko', markersize=7)      
    
# Loading Data
data = scipy.io.loadmat('ex6data1.mat')

#Plotting Data
X = data["X"]
y = data["y"]
m, n = X.shape

plotData(X, y)
plt.show() 
plt.close()

'''
==================== Part 2: Training Linear SVM ====================
'''
print("\nTraining Linear SVM ...\n")
def linearKernel(x1, x2):
    # Ensure that x1 and x2 are column vectors
    x1 = x1.flatten()
    x2 = x2.flatten()
    sim = np.dot(x1, x2.T)
    return sim

def gaussianKernel(x1, x2, sigma=0.1):
    # Ensure that x1 and x2 are column vectors
    x1 = x1.flatten()
    x2 = x2.flatten()
    
    sim = 0
    sim = np.exp(-np.sum(np.power((x1-x2), 2)) / float(2*(sigma**2)))
    return sim

def gaussianKernelGramMatrix(x1, x2, K_function=gaussianKernel, sigma=0.1):
    GramMaxtrix = np.zeros((x1.shape[0], x2.shape[0]))
    
    for i, x1elem in enumerate(x1):
        for j, x2elem in enumerate(x2):
            GramMaxtrix[i, j] = K_function(x1elem, x2elem, sigma)
            
    return GramMaxtrix

'''
Sklearn SVM Reference:
https://scikit-learn.org/stable/modules/svm.html
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
'''
def svmTrain(X, y, C, kernelFunction, tol=1e-3, max_passes=-1, sigma=0.1):
    y = y.flatten()
    
    if kernelFunction == "gaussian":
        clf = svm.SVC(C=C, kernel='precomputed', tol=tol, max_iter=max_passes, verbose=2)
        return clf.fit(gaussianKernelGramMatrix(X, X, sigma=sigma), y)
    else:
        clf = svm.SVC(C=C, kernel=kernelFunction, tol=tol, max_iter=max_passes, verbose=2)
        return clf.fit(X, y)

def visualizeBoundaryLinear(X, y, model):
    w = model.coef_[0]
    b = model.intercept_[0]
    xp = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    yp = -(w[0] * xp + b) / w[1]
    
    plt.plot(xp, yp, 'b-')
    plotData(X, y)
    
# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
C = 1000
model = svmTrain(X, y, C, "linear", 1e-3, 20)
visualizeBoundaryLinear(X, y, model)
plt.show() 
plt.close()

'''
=============== Part 3: Implementing Gaussian Kernel ===============
'''
print('\nEvaluating the Gaussian Kernel ...\n')
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gaussianKernel(x1, x2, sigma)

print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 2 :' \
         '\n\t{:f}\n(this value should be about 0.324652)\n'.format(sim))

'''
=============== Part 4: Visualizing Dataset 2 ================
'''
print('Loading and Visualizing Data ...\n')

# Loading Data
data = scipy.io.loadmat('ex6data2.mat')

#Plotting Data
X = data["X"]
y = data["y"]
m, n = X.shape

plotData(X, y)
plt.show() 
plt.close()

'''
========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
'''
print('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n')
def visualizeBoundary(X, y, model, varargin=0):
    plotData(X, y)
    
    x1plot = np.linspace(X[:,0].min(), X[:,0].max(), 100).T
    x2plot = np.linspace(X[:,1].min(), X[:,1].max(), 100).T
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)
    
    for i in range(X1.shape[1]):
        this_X = np.column_stack((X1[:,i], X2[:,i]))
        vals[:,i] = model.predict(gaussianKernelGramMatrix(this_X, X))
        
    plt.contour(X1, X2, vals, colors="blue", levels=[0,0])

# SVM Parameters
C = 1
sigma = 0.1
model = svmTrain(X, y, C, "gaussian")
#clf = svm.SVC(C=C, kernel='rbf', gamma=6, tol=1e-3, max_iter=20, verbose=2)
#model = clf.fit(X, y)
visualizeBoundary(X, y, model)
plt.show() 
plt.close()

'''
=============== Part 6: Visualizing Dataset 3 ================
'''
print('Loading and Visualizing Data ...\n')
# Loading Data
data = scipy.io.loadmat('ex6data3.mat')

#Plotting Data
X = data["X"]
y = data["y"]
m, n = X.shape

plotData(X, y)
plt.show() 
plt.close()

'''
========== Part 7: Training SVM with RBF Kernel (Dataset 3) =========
'''
def dataset3Params(X, y, Xval, yval):
    sigma = 0.3
    C = 1
    
    C_try = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_try = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    # vector with all predictions from SVM
    predictionErrors = np.zeros((len(C_try)*len(sigma_try),3))
    predictionsCounter = 0
    
    for s in sigma_try:
        for c in C_try:
            model = svmTrain(X, y, c, "gaussian", sigma=s)
            predictions = model.predict(gaussianKernelGramMatrix(Xval, X))
            
            # compute prediction errors on cross-validation set
            predictionErrors[predictionsCounter,0] = np.mean((predictions != yval).astype(int))
            # store corresponding C and sigma
            predictionErrors[predictionsCounter,1] = s      
            predictionErrors[predictionsCounter,2] = c
            
            predictionsCounter = predictionsCounter + 1
    
    ErrorMinIndex = predictionErrors.argmin(axis=0)
    sigma = predictionErrors[ErrorMinIndex[0], 1]
    C = predictionErrors[ErrorMinIndex[0], 2]
    
    return C, sigma
    
Xval = data["Xval"]
yval = data["yval"]
# Try different SVM Parameters here
C, sigma = dataset3Params(X, y, Xval, yval)

model = svmTrain(X, y, C, "gaussian", sigma=sigma)
visualizeBoundary(X, y, model)
plt.show() 
plt.close()
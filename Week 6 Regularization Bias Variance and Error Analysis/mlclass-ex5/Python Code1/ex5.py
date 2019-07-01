# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 13:04:27 2019

@author: Yu Chen
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.optimize import minimize

'''
============ Part 1: Loading and Visualizing Data ============
'''
print("Loading and Visualizing Data ...\n")
def displayData(x, y):
    plt.plot(x, y, 'rx', markersize=10, linewidth=1.5)
    plt.xlabel("Change in water level (x)")
    plt.ylabel("Water flowing out of the dam (y)")         
    
# Loading Data
data = scipy.io.loadmat('ex5data1.mat')

#Plotting Data
X = data["X"]
y = data["y"]
Xval = data["Xval"]
yval = data["yval"]
Xtest = data["Xtest"]
ytest = data["ytest"]
m, n = X.shape

displayData(X, y)
plt.show() 
plt.close()

'''
=========== Part 2: Regularized Linear Regression Cost =============
'''
def linearRegCostFunction(X, y, theta, lambda_reg):
    m = len(y)
    # force to be 2D vector in column
    theta = np.reshape(theta, (-1,y.shape[1]))
    
    J = 0
    grad = np.zeros(len(theta))
    
    H_theta = np.dot(X, theta)
    J = (1./(2*m)) * np.power((H_theta - y) , 2).sum() + (float(lambda_reg)/(2*m)) * np.power(theta[1:theta.shape[0]],2).sum()
    grad = (1./m) * np.dot(X.T, (H_theta - y)) + (float(lambda_reg)/m) * theta
    # unregularize first gradient
    grad_no_regularization = (1./m) * np.dot(X.T, (H_theta - y))
    grad[0] = grad_no_regularization[0]
    
    return J, grad
    
theta = np.array([[1] , [1]])
X_padded = np.column_stack((np.ones((m,1)), X))
J, grad = linearRegCostFunction(X_padded, y, theta, 1)

print('Cost at theta = [1 ; 1]: {:f}\n(this value should be about 303.993192)\n'.format(J))

'''
=========== Part 3: Regularized Linear Regression Gradient =============
'''

print('Gradient at theta = [1 ; 1]:  [{:f}; {:f}] \n(this value should be about [-15.303016; 598.250744])'.format(float(grad[0]), float(grad[1])))

'''
=========== Part 4: Train Linear Regression =============
'''
def trainLinearReg(X, y, lambda_reg):
    # Initialize Theta
    initial_theta = np.zeros((X.shape[1], 1))
    
    # Short hand for cost function to be minimized
    def costFunc(theta):
        return linearRegCostFunction(X, y, theta, lambda_reg)
    
    # Now, costFunction is a function that takes in only one argument
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    maxiter = 200
    results = minimize(costFunc, x0=initial_theta, options={'disp': True, 'maxiter':maxiter}, method="L-BFGS-B", jac=True)

    theta = results["x"]

    return theta

def displayTrainData(x, y, theta):
    plt.plot(x, y, 'rx', markersize=10, linewidth=1.5)
    plt.xlabel("Change in water level (x)")
    plt.ylabel("Water flowing out of the dam (y)")
    plt.plot(X, np.dot(np.column_stack((np.ones((m,1)), X)), theta), '--', linewidth=2)
    
    
#  Train linear regression with lambda = 0
lambda_reg = 0
theta = trainLinearReg(X_padded, y, lambda_reg)

displayTrainData(X, y, theta)
plt.show()  
plt.close()

'''
=========== Part 5: Learning Curve for Linear Regression =============
'''
def learningCurve(X, y, Xval, yval, lambda_reg):
    m = len(y)
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))
    
    for i in range(1, m+1):
        X_train = X[:i]
        y_train = y[:i]
        
        theta = trainLinearReg(X_train, y_train, lambda_reg)
        
        error_train[i-1], grad = linearRegCostFunction(X_train, y_train, theta, 0)
        error_val[i-1], grad = linearRegCostFunction(Xval, yval, theta, 0)
    
    return error_train, error_val

def displayLearningCurve(Etrain, Eval):
    m = len(Etrain)
    p1, p2 = plt.plot(range(m), Etrain, range(m), Eval)
    plt.title('Learning curve for linear regression')
    plt.legend((p1, p2), ('Train', 'Cross Validation'), numpoints=1, handlelength=0.5)
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    
    
lambda_reg = 0
error_train, error_val = learningCurve(np.column_stack((np.ones((m,1)), X)), y, np.column_stack((np.ones((Xval.shape[0], 1)), Xval)), yval, lambda_reg)
displayLearningCurve(error_train, error_val)
plt.show()
plt.close()

print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('  \t{:d}\t\t{:f}\t{:f}\n'.format(i+1, float(error_train[i]), float(error_val[i])))
    
'''
=========== Part 6: Feature Mapping for Polynomial Regression =============
'''
def polyFeatures(X, p):
    X_poly = np.zeros((len(X), p))
    x = X.flatten()
    for i in range(0, p):
        X_poly[:, i] = np.power(x, i+1)
    
    return X_poly

def featureNormalize(X):
    mu = np.mean(X, axis = 0)
    X_norm = X - mu
    
    sigma = np.std(X_norm, axis = 0)
    X_norm = X_norm / sigma
    
    return X_norm, mu, sigma

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)  # Normalize
X_poly = np.column_stack((np.ones((m,1)), X_poly)) # Add Ones

# Map Xtest onto Polynomial Features and Normalize
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
X_poly_test = np.column_stack((np.ones((X_poly_test.shape[0],1)), X_poly_test)) # Add Ones

# Map Xval onto Polynomial Features and Normalize
X_poly_val = polyFeatures(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = np.column_stack((np.ones((X_poly_val.shape[0],1)), X_poly_val)) # Add Ones

print('Normalized Training Example 1:')
print(*X_poly[0, :], sep="\n")

'''
=========== Part 7: Learning Curve for Polynomial Regression =============
'''
def plotFit(min_x, max_x, mu, sigma, theta, p):
    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.array(np.arange(min_x - 15, max_x + 25, 0.05)) # 1D vector
    
    # Map the X values 
    X_poly = polyFeatures(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly / sigma
    X_poly = np.column_stack((np.ones((x.shape[0],1)), X_poly))
    plt.plot(x, np.dot(X_poly, theta), '--', linewidth=2)
    plt.axis([min_x-15, max_x+15, -60, 120])

lambda_reg = 0;
theta = trainLinearReg(X_poly, y, lambda_reg)

plt.figure(1)
displayData(X, y)
plotFit(min(X), max(X), mu, sigma, theta, p)
plt.title ('Polynomial Regression Fit (lambda = {:f})'.format(lambda_reg))

plt.figure(2)
error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda_reg)
displayLearningCurve(error_train, error_val)
plt.title('Polynomial Regression Learning Curve (lambda = {:f})'.format(lambda_reg))
plt.show()
plt.close()

print('Polynomial Regression (lambda = {:f})\n\n'.format(lambda_reg))
print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('  \t{:d}\t\t{:f}\t{:f}\n'.format(i+1, float(error_train[i]), float(error_val[i])))
    
'''
=========== Part 8: Validation for Selecting Lambda =============
'''
def validationCurve(X, y, Xval, yval):
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    error_train = np.zeros((len(lambda_vec), 1))
    error_val = np.zeros((len(lambda_vec), 1))
    
    for i in range(len(lambda_vec)):
        lambda_reg = lambda_vec[i]

        # learn theta parameters with current lambda value
        theta = trainLinearReg(X, y, lambda_reg)

        # fill in error_train[i] and error_val[i]
        #   note that for error computation, we set lambda = 0 in the last argument
        error_train[i], grad = linearRegCostFunction(X,    y,    theta, 0)
        error_val[i], grad   = linearRegCostFunction(Xval, yval, theta, 0)
        
    return lambda_vec, error_train, error_val
    
lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

p1, p2 = plt.plot(lambda_vec, error_train, lambda_vec, error_val)
plt.legend((p1, p2), ('Train', 'Cross Validation'))
plt.xlabel('lambda')
plt.ylabel('Error')
plt.axis([0, 10, 0, 20])
plt.show()
plt.close()

print('lambda\t\tTrain Error\tValidation Error\n')
for i in range(len(lambda_vec)):
	print(' {:f}\t{:f}\t{:f}\n'.format(float(lambda_vec[i]), float(error_train[i]), float(error_val[i])))

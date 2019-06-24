# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:46:34 2019

@author: Yu Chen
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.optimize as op

''' Initialization and Setup Parameter '''
input_layer_size = 400 # 20x20 Input Images of Digits
num_labels = 10 # 10 labels, from 1 to 10 (note that we have mapped 0 to label 10)

'''
============ Part 1: Loading and Visualizing Data ============
'''
def displayData(X, example_width=None):
    if example_width == None:
            example_width = int(np.round(np.sqrt(np.shape(X)[1])))
        
    m, n = np.shape(X)
    example_height = int(n/example_width)
        
    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m/display_rows))
        
    # Beteen images padding
    pad = 1
        
    # Setup blank display
    display_array = -np.ones((pad+display_rows*(example_height + pad), \
                pad + display_cols * (example_width + pad)))
    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m:
                break
            # Copy the patch and get the max value of the patch
            max_val = np.max(np.abs(X[curr_ex,:]))
            initial_x = pad + j * (example_height + pad)
            initial_y =pad + i * (example_width + pad)
            display_array[initial_x:initial_x+example_height, \
                        initial_y:initial_y+example_width] = \
                         X[curr_ex, :].reshape(example_height, example_width)\
                         / max_val
            curr_ex += 1
        if curr_ex > m:
            break
    
    
    # Display image
    img = scipy.misc.toimage(display_array)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)  
    plt.show()          

print("Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples \n")
# Loading Data
data = scipy.io.loadmat('ex3data1.mat')

#Plotting Data
X = data["X"]
y = data["y"]
m, n = X.shape

# Important to performance: changes the dimension from (m,1) to (m,)
y = y.flatten()

# Randomly select 100 data points to display
rand_index = np.random.permutation(m)
sel = X[rand_index[:100],:]

displayData(sel)

'''
============ Part 2: Compute Cost and Gradient ============
'''
def sigmoid(z):
    #g = np.zeros(zi.shape[0])
    g = 1/(1 + np.exp(-z))
    return g
     
def lrCostFunction(theta, X, y, lambda_reg, return_grad=False):
    m = y.size
    J = 0
    grad = np.zeros(theta.shape)
    
    one = y * np.transpose(np.log(sigmoid(np.dot(X,theta))))
    two = (1 - y) * np.transpose(np.log(1 - sigmoid(np.dot(X,theta))))
    reg = (float(lambda_reg) / (2 * m)) * np.power(theta[1:theta.shape[0]],2).sum()
    J = -(1./m)*(one + two).sum() + reg
    
    grad = (1./m) * np.dot(sigmoid(np.dot(X,theta)).T - y, X).T + (float(lambda_reg) / m ) * theta
    
    grad_no_regularization = (1./m) * np.dot(sigmoid(np.dot(X,theta)).T - y, X).T
    grad[0] = grad_no_regularization[0]
    
    sys.stdout.write("Cost: %f   \r" % (J) )
    sys.stdout.flush()
    if return_grad:
        return J, grad.flatten()
    else:
        return J

def oneVSAll(X, y, num_labels, lambda_reg):
    m, n = X.shape
    all_theta = np.zeros((num_labels, n + 1))
    X = np.column_stack((np.ones((m,1)), X))
    
    for c in range(num_labels):
        initial_theta = np.zeros((n + 1, 1))
        print("Training {:d} out of {:d} categories...".format(c+1, num_labels))
        myargs = (X, (y%10==c).astype(int), lambda_reg, True)
        theta = op.minimize(lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':13}, method="Newton-CG", jac=True)
        all_theta[c,:] = theta["x"]
    
    return all_theta

def predictOneVSAll(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]
    p = np.zeros((m, 1))
    
    X = np.column_stack((np.ones((m,1)), X))
    
    p = np.argmax(sigmoid(np.dot(X,all_theta.T)), axis=1)
    return p

print('Training One-vs-All Logistic Regression...') 

lambda_reg = 0.1
all_theta = oneVSAll(X, y, num_labels, lambda_reg) 
pred = predictOneVSAll(all_theta, X)

print('Training Set Accuracy: {:f}'.format((np.mean(pred == y%10)*100)))
print('Training Set Accuracy for 1:  {:f}'.format(np.mean(pred[500:1000]  == y.flatten()[500:1000]%10)  * 100))
print('Training Set Accuracy for 2:  {:f}'.format(np.mean(pred[1000:1500] == y.flatten()[1000:1500]%10) * 100))
print('Training Set Accuracy for 3:  {:f}'.format(np.mean(pred[1500:2000] == y.flatten()[1500:2000]%10) * 100))
print('Training Set Accuracy for 4:  {:f}'.format(np.mean(pred[2000:2500] == y.flatten()[2000:2500]%10) * 100))
print('Training Set Accuracy for 5:  {:f}'.format(np.mean(pred[2500:3000] == y.flatten()[2500:3000]%10) * 100))
print('Training Set Accuracy for 6:  {:f}'.format(np.mean(pred[3000:3500] == y.flatten()[3000:3500]%10) * 100))
print('Training Set Accuracy for 7:  {:f}'.format(np.mean(pred[3500:4000] == y.flatten()[3500:4000]%10) * 100))
print('Training Set Accuracy for 8:  {:f}'.format(np.mean(pred[4000:4500] == y.flatten()[4000:4500]%10) * 100))
print('Training Set Accuracy for 9:  {:f}'.format(np.mean(pred[4500:5000] == y.flatten()[4500:5000]%10) * 100))
print('Training Set Accuracy for 10: {:f}'.format(np.mean(pred[0:500]     == y.flatten()[0:500]%10)     * 100))

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:32:01 2019

@author: Yu Chen
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.optimize import minimize
from decimal import Decimal

''' Initialization and Setup Parameter '''
input_layer_size = 400 # 20x20 Input Images of Digits
hidden_layer_size = 25 # 25 hidden units
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

print("Loading and Visualizing Data ...\n")
# Loading Data
data = scipy.io.loadmat('ex4data1.mat')

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
============ Part 2: Loading Parameters ============
'''
print('\nLoading Saved Neural Network Parameters ...\n')
weight = scipy.io.loadmat('ex4weights.mat')
Theta1 = weight["Theta1"]
Theta2 = weight["Theta2"]

'''
============ Part 3: Compute Cost (Feedforward) ============
'''
print('\nFeedforward Using Neural Network ...\n')
def sigmoid(z):
    #g = np.zeros(zi.shape[0])
    g = 1/(1 + np.exp(-z))
    return g

def sigmoidGradient(z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g * (1 - g)

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg):
    '''
    NNCOSTFUNCTION Implements the neural network cost function for a two layer neural network which performs classification
    Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices for our 2 layer neural network
    '''
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1), order='F')
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1), order='F')
    
    m, n = X.shape
    J = 0;
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    
    ''' Part 1: Feedforward the neural network and return the cost in the cost '''
    X = np.column_stack((np.ones((m,1)), X))
    a2 = sigmoid(np.dot(X,Theta1.T))
    a2 = np.column_stack((np.ones((a2.shape[0],1)), a2))
    a3 = sigmoid(np.dot(a2,Theta2.T))
    
    # COST FUNCTION CALCULATION
    # NONREGULARIZED COST FUNCTION
    # recode labels as vectors containing only values 0 or 1
    labels = y
    # set y to be matrix of size m x k
    y = np.zeros((m,num_labels))
    # for every label, convert it into vector of 0s and a 1 in the appropriate position
    for i in range(m):
    	y[i, labels[i]-1] = 1
    
    cost = 0
    for i in range(m):
    	cost += np.sum(y[i] * np.log(a3[i]) + (1 - y[i]) * np.log(1 - a3[i]))
    J = -(1.0/m) * cost
    
    # REGULARIZED COST FUNCTION
    sumOfTheta1 = np.sum(np.sum(Theta1[:,1:]**2))
    sumOfTheta2 = np.sum(np.sum(Theta2[:,1:]**2))
    J = J + ((lambda_reg / (2.0 * m)) * (sumOfTheta1 + sumOfTheta2))
    
    ''' Part 2: Implement the backpropagation algorithm to compute the gradients Theta1_grad and Theta2_grad '''
    # BACKPROPAGATION
    tridelta_1 = 0
    tridelta_2 = 0
    
    # for each training example
    for t in range(m):
        ## step 1: perform forward pass
        x = X[t]
        a2 = sigmoid(np.dot(x,Theta1.T))
        a2 = np.concatenate((np.array([1]), a2))
        a3 = sigmoid(np.dot(a2,Theta2.T))
        
        ## step 2: for each output unit k in layer 3, set delta_{k}^{(3)}
        delta3 = np.zeros((num_labels))
        for k in range(num_labels):
            y_k = y[t, k]
            delta3[k] = a3[k] - y_k
        
        ## step 3: for the hidden layer l=2, set delta2 = Theta2' * delta3 .* sigmoidGradient(z2)
        delta2 = (np.dot(Theta2[:,1:].T, delta3).T) * sigmoidGradient(np.dot(x, Theta1.T))
        
        ## step 4: accumulate gradient from this example
        tridelta_1 += np.outer(delta2, x)
        tridelta_2 += np.outer(delta3, a2)
        
    # step 5: obtain gradient for neural net cost function by dividing the accumulated gradients by m
    Theta1_grad = tridelta_1 / m
    Theta2_grad = tridelta_2 / m
    
    ''' Part 3: Implement regularization with the cost function and gradients '''
    # REGULARIZATION FOR GRADIENT
    Theta1_grad_unregularized = np.copy(Theta1_grad)
    Theta2_grad_unregularized = np.copy(Theta2_grad)
    Theta1_grad += (float(lambda_reg)/m)*Theta1
    Theta2_grad += (float(lambda_reg)/m)*Theta2
    Theta1_grad[:,0] = Theta1_grad_unregularized[:,0]
    Theta2_grad[:,0] = Theta2_grad_unregularized[:,0]
    
    # Unroll gradients
    grad = np.concatenate((Theta1_grad.reshape(Theta1_grad.size, order='F'), Theta2_grad.reshape(Theta2_grad.size, order='F')))

    return J, grad

# # Weight regularization parameter (we set this to 0 here).
lambda_reg = 0
nn_params = np.concatenate((Theta1.reshape(Theta1.size, order='F'), Theta2.reshape(Theta2.size, order='F')))
J, grad= nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)
print('Training Set Accuracy: {:f}\n(this value should be about 0.287629)'.format(J))

'''
=============== Part 4: Implement Regularization ===============
'''
print('\nChecking Cost Function (w/ Regularization) ... \n')

# Weight regularization parameter (we set this to 1 here).
lambda_reg = 1
J, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)
print('Cost at parameters (loaded from ex4weights): {:f}\n(this value should be about 0.383770)'.format(J))

'''
================ Part 5: Sigmoid Gradient  ================
'''
print('\nEvaluating sigmoid gradient...\n')

g = sigmoidGradient(np.array([1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [1, -0.5, 0, 0.5, 1]:')
print(*g, sep="\n")

'''
================ Part 6: Initializing Pameters ================
'''
print('\nInitializing Neural Network Parameters ...\n')
def randInitializeWeights(L_in, L_out):
    W = np.zeros((L_out, 1 + L_in))
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in) * (2*epsilon_init) - epsilon_init

    return W

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
# Unroll parameters
initial_nn_params = np.concatenate((initial_Theta1.reshape(initial_Theta1.size, order='F'), initial_Theta2.reshape(initial_Theta2.size, order='F')))

'''
=============== Part 7: Implement Backpropagation ===============
'''
print('\nChecking Backpropagation... \n')
def computeNumericalGradient(J, theta):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    
    for p in range(theta.size):
        # Set perturbation vector
        perturb.reshape(perturb.size, order="F")[p] = e
        loss1, grad1 = J(theta - perturb)
        loss2, grad2 = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad.reshape(numgrad.size, order="F")[p] = (loss2 - loss1) / (2*e)
        perturb.reshape(perturb.size, order="F")[p] = 0

    return numgrad

def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros((fan_out, 1 + fan_in))
    W = np.reshape(np.sin(range(W.size)), W.shape) / 10

    return W

def checkNNGradients(lambda_reg=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    X  = debugInitializeWeights(m, input_layer_size - 1)
    y  = 1 + np.mod(range(m), num_labels).T
    
    # Unroll parameters
    nn_params = np.concatenate((Theta1.reshape(Theta1.size, order='F'), Theta2.reshape(Theta2.size, order='F')))
    
    # Short hand for cost function
    def costFunc(p):
        return nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)

    cost, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)
    
    # Visually examine the two gradient computations.  The two columns
    # you get should be very similar. 
    # code from http://stackoverflow.com/a/27663954/583834
    fmt = '{:<25}{}'
    print(fmt.format('Numerical Gradient', 'Analytical Gradient'))
    for numerical, analytical in zip(numgrad, grad):
        print(fmt.format(numerical, analytical))

    print('The above two columns you get should be very similar.\n' '(Left Col.: Your Numerical Gradient, Right Col.: Analytical Gradient)')
    diff = Decimal(np.linalg.norm(numgrad-grad))/Decimal(np.linalg.norm(numgrad+grad))
    print('If your backpropagation implementation is correct, then \n' \
             'the relative difference will be small (less than 1e-9). \n' \
             '\nRelative Difference: {:.10E}'.format(diff))

checkNNGradients()

'''
=============== Part 8: Implement Regularization ===============
'''
print('\nChecking Backpropagation (w/ Regularization) ... \n')

#  Check gradients by running checkNNGradients
lambda_reg = 3
checkNNGradients(lambda_reg)

# Also output the costFunction debugging values
debug_J, grad  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)

print('\n\nCost at (fixed) debugging parameters (w/ lambda_reg = 3): {:f} ' \
         '\n(this value should be about 0.576051)\n\n'.format(debug_J))

'''
=================== Part 9: Training NN ===================
'''
print('\nTraining Neural Network... \n')
#   http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
maxiter = 20
lambda_reg = 0.1
myargs = (input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)
results = minimize(nnCostFunction, x0=nn_params, args=myargs, options={'disp': True, 'maxiter':maxiter}, method="L-BFGS-B", jac=True)

nn_params = results["x"]

# Obtain Theta1 and Theta2 back from nn_params
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1), order='F')
Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1), order='F')

'''
================= Part 10: Visualize Weights =================
'''
print('\nVisualizing Neural Network... \n')
displayData(Theta1[:, 1:])

'''
================= Part 11: Implement Predict =================
'''
def predict(Theta1, Theta2, X):
    # turns 1D X array into 2D
    if X.ndim == 1:
        X = np.reshape(X, (-1, X.shape[0]))
        
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    p = np.zeros((m,1))
    
    X = np.column_stack((np.ones((m,1)), X))
    a2 = sigmoid(np.dot(X,Theta1.T))
    a2 = np.column_stack((np.ones((a2.shape[0],1)), a2))
    a3 = sigmoid(np.dot(a2,Theta2.T))
    
    p = np.argmax(a3, axis=1)
    return p + 1

pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: {:f}'.format((np.mean(pred == y)*100)))






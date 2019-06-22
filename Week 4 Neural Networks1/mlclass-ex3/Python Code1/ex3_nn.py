# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:46:34 2019

@author: Yu Chen
"""

import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.optimize as op

''' Initialization and Setup Parameter '''
input_layer_size = 400 # 20x20 Input Images of Digits
hidden_layer_size = 25 # 25 hidden units
num_labels = 10 # 10 labels, from 1 to 10 (note that we have mapped 0 to label 10)

'''
============ Part 1: Loading and Visualizing Data ============
'''
def displayData(X, example_width=None):
    plt.close()
    plt.figure()
    
    if X.ndim == 1:
        X = np.reshape(X, (-1, X.shape[0]))
        
    if not example_width or not 'example_width' in locals():
        example_width = int(round(math.sqrt(X.shape[1])))
    
    plt.set_cmap("")
    
    m, n = X.shape
    example_height = n / example_width
    
    display_rows = int(math.floor(math.sqrt(m)))
    display_cols = int(math.ceil(m / display_rows))
    
    pad = 1
    
    display_array = - np.ones((pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad)))
    
    curr_ex = 1
    for j in range(1, display_rows+1):
        for i in range(1, display_cols+1):
            if curr_ex > m:
                break
            
            max_val = max(abs(X[curr_ex-1, :]))
            rows = pad + (j - 1) * (example_height + pad) + np.array(range(example_height))
            cols = pad + (i - 1) * (example_width  + pad) + np.array(range(example_width ))
            
            display_array[rows[0]:rows[-1]+1 , cols[0]:cols[-1]+1] = np.reshape(X[curr_ex-1, :], (example_height, example_width), order="F") / max_val
            curr_ex += 1
        
        if curr_ex > m:
            break
        
    h = plt.imshow(display_array, vmin=-1, vmax=1)
    
    plt.axis('off')
    plt.show(block=False)
    
    return h, display_array        

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
============ Part 2: Loading Parameters ============
'''
print('Loading Saved Neural Network Parameters ...')
weight = scipy.io.loadmat('ex3weights.mat')
Theta1 = weight["Theta1"]
Theta2 = weight["Theta2"]

'''
============ Part 3: Implement Predict ============
'''
def sigmoid(z):
    #g = np.zeros(zi.shape[0])
    g = 1/(1 + np.exp(-z))
    return g

def predict(Theta1, Theta2, X):
    # turns 1D X array into 2D
    if X.ndim == 1:
        X = np.reshape(X, (-1,X.shape[0]))
        
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

rp = np.random.permutation(m)

for i in range(m):

    # Display 
    print('Displaying Example Image')
    displayData(X[rp[i], :])

    pred = predict(Theta1, Theta2, X[rp[i], :])
    print('Neural Network Prediction: {:d} (digit {:d})'.format(pred[0], (pred%10)[0]))




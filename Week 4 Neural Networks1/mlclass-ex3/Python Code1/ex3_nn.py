# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:46:34 2019

@author: Yu Chen
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

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




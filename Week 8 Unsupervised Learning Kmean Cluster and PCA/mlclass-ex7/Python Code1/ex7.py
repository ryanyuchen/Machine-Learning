# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:28:47 2019

@author: Yu Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.io
import sys

'''
================= Part 1: Find Closest Centroids ====================
'''
print("Finding closest centroids.\n\n")
def findClosestCentroids(X, centroids):
    K = centroids.shape[0]
    m, n = X.shape
    
    idx = np.zeros((m, 1), dtype=np.int8)
    distance = np.zeros((K, 1))
    
    for i in range(m):
        for k in range(K):
            coorddiff = X[i, :] - centroids[k, :]
            distance[k] = np.sqrt(coorddiff.dot(coorddiff.T))
        idx[i] = np.argmin(distance, axis=0)
        
    return idx

# Loading Data
data = scipy.io.loadmat('ex7data2.mat')
X = data["X"]

# Select an initial set of centroids
K = 3 # 3 Centroids
initial_centroids = np.array( [[3, 3], [6, 2], [8, 5]] )

idx = findClosestCentroids(X, initial_centroids)

print('Closest centroids for the first 3 examples: \n')
print(*idx[:3], sep=' ')
# adjusted next string for python's 0-indexing
print('\n(the closest centroids should be 0, 2, 1 respectively)\n')

'''
===================== Part 2: Compute Means =========================
'''
print("\nComputing centroids means.\n\n")
def computeCentroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    
    for k in range(K):
        points = np.where(idx == k)[0]
        centroids[k] = np.mean(X[points,:], axis=0)
    
    return centroids
    
#  Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K)

print('Centroids computed after initial finding of closest centroids: \n')
print(*centroids, sep='\n')
print('\n(the centroids should be\n')
print('   [ 2.428301 3.157924 ]\n')
print('   [ 5.813503 2.633656 ]\n')
print('   [ 7.119387 3.616684 ]\n\n')

'''
=================== Part 3: K-Means Clustering ======================
'''
print('\nRunning K-Means clustering on example dataset.\n\n')
def drawLine(p1, p2, **kwargs):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)

def hsv(n=63):
    return colors.hsv_to_rgb(np.column_stack([np.linspace(0, 1, n+1), np.ones(((n+1), 2))]))

def plotDataPoints(X, idx, K):
    palette = hsv(K)
    colors = np.array([palette[int(i)] for i in idx])
    
    plt.scatter(X[:,0], X[:,1], s=75, facecolors='none', edgecolors=colors)
    
def plotProgresskMeans(X, centroids, previous, idx, K, i):
    # Plot the examples
    plotDataPoints(X, idx, K)
    
    # Plot the centroids as black x's
    plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=400, c='k', linewidth=1)
    
    # Plot the history of the centroids with lines
    for j in range(centroids.shape[0]):
        drawLine(centroids[j, :], previous[j, :], c='b')

    # Title
    plt.title('Iteration number {:d}'.format(i+1))
     
def runKMeans(X, initial_centroids, max_iters, plot_progress=False):
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m, 1))
    
    # if plotting, set up the space for interactive graphs
    # http://stackoverflow.com/a/4098938/583834
    # http://matplotlib.org/faq/usage_faq.html#what-is-interactive-mode
    if plot_progress:
        plt.close()
        plt.ion()
    
    # Run K-Means
    for i in range(max_iters):
        
        # Output progress
        sys.stdout.write('\rK-Means iteration {:d}/{:d}...'.format(i+1, max_iters))
        sys.stdout.flush()
        
        # For each example in X, assign it to the closest centroid
        idx = findClosestCentroids(X, centroids)
        
        # Optionally, plot progress here
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            input('Press enter to continue.')
        
        # Given the memberships, compute new centroids
        centroids = computeCentroids(X, idx, K)

    # Hold off if we are plotting progress
    print('\n')

    # if plot_progress:
    #     plt.hold(False)

    return centroids, idx

# Settings for running K-Means
K = 3
max_iters = 10
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

centroids, idx = runKMeans(X, initial_centroids, max_iters, True)
print('\nK-Means Done.\n')

'''
============= Part 4: K-Means Clustering on Pixels ===============
'''
print('\nRunning K-Means clustering on pixels from an image.\n\n')
def KMeansInitCentroids(X, K):
    centroids = np.zeros((K, X.shape[1]))
    
    # Initialize the centroids to be random examples
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    # Take the first K examples as centroids
    centroids = X[randidx[:K], :]

    return centroids

#  Load an image of a bird
mat = scipy.io.loadmat('bird_small.mat')
A = mat["A"]

A = A / 255.0 # Divide by 255 so that all values are in the range 0 - 1
img_size = A.shape

# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X that we will use K-Means on.
X = A.reshape(img_size[0] * img_size[1], 3, order='F').copy()

# Settings for running K-Means
K = 16
max_iters = 10

# Initialize the centroids randomly. 
initial_centroids = KMeansInitCentroids(X, K)
# Run K-Means
centroids, idx = runKMeans(X, initial_centroids, max_iters)

'''
================= Part 5: Image Compression ======================
'''
print('\nApplying K-Means to compress an image.\n\n')

# Find closest cluster members
idx = findClosestCentroids(X, centroids)

X_recovered = centroids[idx,:]
# Reshape the recovered image into proper dimensions
X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3, order='F')

# Display the original image
plt.close()
plt.subplot(1, 2, 1)
plt.imshow(A) 
plt.title('Original')

# Display compressed image side by side
plt.subplot(1, 2, 2)
plt.imshow(X_recovered)
plt.title( 'Compressed, with {:d} colors.'.format(K) )
plt.show(block=False)


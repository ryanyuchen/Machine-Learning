# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:29:46 2019

@author: Yu Chen
"""

import numpy as np
import scipy.linalg as linalg
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import scipy.io
import sys

'''
================== Part 1: Load Example Dataset  ===================
'''
print("Visualizing example dataset for PCA.\n\n")

# Loading Data
data = scipy.io.loadmat('ex7data1.mat')
X = data["X"]

plt.plot(X[:,0], X[:,1], 'bo')
plt.axis([0.5, 6.5, 2, 8])
plt.show()

'''
=============== Part 2: Principal Component Analysis ===============
'''
print("\nRunning PCA on example dataset.\n\n")
def featureNormalize(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    
    # code below uses python np.std(..., ddof=1) following
    # http://stackoverflow.com/a/27600240/583834
    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm = X_norm / sigma
    
    return X_norm, mu, sigma

def pca(X):
    m, n = X.shape
    U = np.zeros(n)
    S = np.zeros(n)
    
    '''
    Reference:
    https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.linalg.diagsvd.html
    '''
    Sigma = (1/m) * np.dot(X.T, X)
    U, S, V = linalg.svd(Sigma)
    S = linalg.diagsvd(S, len(S), len(S))
    
    return U, S

def drawLine(p1, p2, **kwargs):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)
    
X_norm, mu, sigma = featureNormalize(X)

U, S = pca(X_norm)

drawLine(mu, mu + 1.5 * S[0,0] * U[:,0].T, c='k', linewidth=2)
drawLine(mu, mu + 1.5 * S[1,1] * U[:,1].T, c='k', linewidth=2)
plt.show()

print('Top eigenvector: \n')
print(' U(:,1) = {:f} {:f} \n'.format(U[0,0], U[1,0]))
print('(you should expect to see -0.707107 -0.707107)')

'''
=================== Part 3: Dimension Reduction ===================
'''
print('\nDimension reduction on example dataset.\n\n')
def projectData(X, U, K):
    Z = np.zeros((X.shape[0], K))
    Ureduce = U[:, :K]
    Z = X.dot(Ureduce)
    return Z

def recoverData(Z, U, K):
    X_rec = np.zeros((Z.shape[0], U.shape[0]))
    Ureduce = U[:, :K]
    X_rec = Z.dot(Ureduce.T)
    return X_rec
    
plt.plot(X_norm[:,0], X_norm[:,1], 'bo')
plt.axis([-4, 3, -4, 3])
plt.show()

#  Project the data onto K = 1 dimension
K = 1
Z = projectData(X_norm, U, K)
print('Projection of the first example: {:s}\n'.format(str(Z[0])))
print('(this value should be about 1.481274)\n')

X_rec  = recoverData(Z, U, K)
print('Approximation of the first example: {:f} {:f}\n'.format(X_rec[0, 0], X_rec[0, 1]))
print('(this value should be about  -1.047419 -1.047419)\n')

#  Draw lines connecting the projected points to the original points
plt.plot(X_norm[:,0], X_norm[:,1], 'bo')
plt.scatter(X_rec[:, 0], X_rec[:, 1], s=75, facecolors='none', edgecolors='r')
for i in range(X_norm.shape[0]):
    drawLine(X_norm[i,:], X_rec[i,:], linestyle='--', color='k', linewidth=1)
plt.show()

'''
=============== Part 4: Loading and Visualizing Face Data =============
'''
print('\nLoading face dataset.\n\n')
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

# Loading Data
data = scipy.io.loadmat('ex7faces.mat')
X = data["X"]
displayData(X[:100,:])

'''
=========== Part 5: PCA on Face Data: Eigenfaces  ===================
'''
print('\nRunning PCA on face dataset.\n')
print('(this mght take a minute or two ...)\n\n')

X_norm, mu, sigma = featureNormalize(X)
U, S = pca(X_norm)
displayData(U[:,:36].T)

'''
============= Part 6: Dimension Reduction for Faces =================
'''
print('\nDimension reduction for face dataset.\n\n')

K = 100
Z = projectData(X_norm, U, K)

print('The projected data Z has a size of: ')
print('{:d} {:d}'.format(Z.shape[0], Z.shape[1]))

'''
==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
'''
print('\nVisualizing the projected (reduced dimension) faces.\n\n')

K = 100;
X_rec  = recoverData(Z, U, K)

# Display normalized data
plt.subplot(1, 2, 1)
displayData(X_norm[:100,:])
plt.title('Original faces')

# Display reconstructed data from only k eigenfaces
plt.subplot(1, 2, 2)
displayData(X_rec[:100,:])
plt.title('Recovered faces')

'''
=== Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
'''
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

def computeCentroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    
    for k in range(K):
        points = np.where(idx == k)[0]
        centroids[k] = np.mean(X[points,:], axis=0)
    
    return centroids

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

def KMeansInitCentroids(X, K):
    centroids = np.zeros((K, X.shape[1]))
    
    # Initialize the centroids to be random examples
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    # Take the first K examples as centroids
    centroids = X[randidx[:K], :]

    return centroids

# A = double(imread('bird_small.png'));
mat = scipy.io.loadmat('bird_small.mat')
A = mat["A"]

A = A / 255.0
img_size = A.shape
X = A.reshape(img_size[0] * img_size[1], 3, order='F').copy()
K = 16 
max_iters = 10
initial_centroids = KMeansInitCentroids(X, K)
centroids, idx = runKMeans(X, initial_centroids, max_iters)

#  Sample 1000 random indexes (since working with all the data is
#  too expensive. If you have a fast computer, you may increase this.
#  use flatten(). otherwise, Z[sel, :] yields array w shape [1000,1,2]
sel = np.floor(np.random.rand(1000, 1) * X.shape[0]).astype(int).flatten()

#  Setup Color Palette
palette = hsv(K)
colors = np.array([palette[int(i)] for i in idx[sel]])

# Visualize the data and centroid memberships in 3D
# https://matplotlib.org/2.0.0/mpl_toolkits/mplot3d/tutorial.html
fig1 = plt.figure(1)
ax = Axes3D(fig1)
ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], s=100, c=colors)
plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')
plt.show(block=False)

'''
=== Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
'''

# Subtract the mean to use PCA
X_norm, mu, sigma = featureNormalize(X)

# PCA and project the data to 2D
U, S = pca(X_norm)
Z = projectData(X_norm, U, 2)

# Plot in 2D
fig2 = plt.figure(2)
plotDataPoints(Z[sel, :], idx[sel], K)
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction');
plt.show(block=False)






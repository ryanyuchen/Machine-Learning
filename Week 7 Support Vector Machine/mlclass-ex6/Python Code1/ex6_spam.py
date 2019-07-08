# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:37:47 2019

@author: Yu Chen
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import re
from nltk import PorterStemmer
from sklearn import svm

'''
==================== Part 1: Email Preprocessing ====================
'''
print("\nPreprocessing sample email (emailSample1.txt)\n")
def readFile(filename):
    try:
        with open(filename, 'r') as of:
            file_contents = of.read()
    except:
        file_contents = ''
        print('Unable to open {:s}'.format(filename))
        
    return file_contents

def getVocabList():
    with open('vocab.txt', 'r') as vf:
        vocabList = {}
        for line in vf.readlines():
            i, word = line.split()
            vocabList[word] = int(i)
    return vocabList

'''
Reference:
https://www.geeksforgeeks.org/python-stemming-words-with-nltk/
'''
def processEmail(email_contents):
    vocabList = getVocabList()
    word_index = []
    
    email_contents = email_contents.lower()
    #Strip all heml
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    #Handle URLS
    email_contents = re.sub('(http|https)://[^\s]*', 'heepaddr', email_contents)
    #Handle Email Addresses
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
    #Handle $ sign
    email_contents = re.sub('[$]+', 'dollar', email_contents)
    
    # ========================== Tokenize Email ===========================
    print('\n==== Processed Email ====\n\n')
    # Process file
    l = 0
    email_contents = re.split(r'[@$/#.-:&\*\+=\[\]?!(){},\'\'\">_<;%\s\n\r\t]+', email_contents)
    
    for token in email_contents:
        # Remove any non alphanumeric characters
        token = re.sub('[^a-zA-Z0-9]', '', token)
        # Stem the word 
        token = PorterStemmer().stem(token.strip())
        
        if len(token) < 1:
           continue
       
        idx = vocabList[token] if token in vocabList else 0
        if idx > 0:
            word_index.append(idx)
        # Print to screen, ensuring that the output lines are not too long
        if l + len(token) + 1 > 78:
            print("")
            l = 0
        print('{:s}'.format(token)),
        l = l + len(token) + 1
    
    print('\n\n=========================\n')
    return word_index
     
# Extract Features
file_contents = readFile('emailSample1.txt')
word_index  = processEmail(file_contents)

# Print Stats
print('Word Indices: ')
print(*word_index, sep=' ')
print('\n\n')

'''
==================== Part 2: Feature Extraction ====================
'''
print("\nExtracting features from sample email (emailSample1.txt)\n")
def emailFeatures(word_indices):
    n = 1899
    x = np.zeros((n, 1))
    
    for i in word_indices:
        x[i] = 1
    
    return x

features = emailFeatures(word_index)
print('Length of feature vector: {:d}'.format(len(features)))
print('Number of non-zero entries: {:d}'.format(np.sum(features>0)))

'''
=========== Part 3: Train Linear SVM for Spam Classification ========
'''
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

data = scipy.io.loadmat('spamTrain.mat')
X = data["X"]
y = data["y"]
y = y.flatten()

print("\nTraining Linear SVM (Spam Classification)\n")
print("(this may take 1 to 2 minutes) ...\n")

C = 0.1
model = svmTrain(X, y, C, "linear")
p = model.predict(X)
print('Training Accuracy: {:f}\n'.format(np.mean((p == y).astype(int)) * 100))

'''
=================== Part 4: Test Spam Classification ================
'''
data = scipy.io.loadmat('spamTest.mat')
Xtest = data["Xtest"]
ytest = data["ytest"]
ytest = ytest.flatten()

print("\nEvaluating the trained Linear SVM on a test set ...\n")

p = model.predict(Xtest)
print('Training Accuracy: {:f}\n'.format(np.mean((p == ytest).astype(int)) * 100))

'''
================= Part 5: Top Predictors of Spam ====================
'''
# Sort the weights and obtain the vocabulary list
w = model.coef_[0]

# from http://stackoverflow.com/a/16486305/583834
# reverse sorting by index
index = w.argsort()[::-1][:15]
vocabList = sorted(getVocabList().keys())

print('\nTop predictors of spam: \n');
for idx in index: 
    print('{:s} ({:f})'.format(vocabList[idx], float(w[idx])))

'''
=================== Part 6: Try Your Own Emails =====================
'''
filename = 'spamSample1.txt'
file_contents = readFile(filename)
word_index = processEmail(file_contents)
x = emailFeatures(word_index)
p = model.predict(x.T)

print('\nProcessed {:s}\n\nSpam Classification: {:s}\n'.format(filename, str(p[0])))
print('(1 indicates spam, 0 indicates not spam)\n\n')

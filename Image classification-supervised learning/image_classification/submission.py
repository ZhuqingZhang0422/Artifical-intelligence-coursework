import collections
import numpy as np

############################################################
# Problem 4.1

def runKMeans(k,patches,maxIter):
    """
    Runs K-means to learn k centroids, for maxIter iterations.
    
    Args:
      k - number of centroids.
      patches - 2D numpy array of size patchSize x numPatches
      maxIter - number of iterations to run K-means for

    Returns:
      centroids - 2D numpy array of size patchSize x k
    """
    # This line starts you out with randomly initialized centroids in a matrix 
    # with patchSize rows and k columns. Each column is a centroid.
    # BEGIN_YOUR_CODE
    n = patches.shape[0]
    patch_t = np.array(patches)
    # Initilize the centers
    center = np.random.randn(n,k)
    num = patches.shape[1]
    patch_cluster = np.zeros(num)
    for i in range(maxIter):
        counts = np.zeros(k)
        for col in range(num):
            temp = center - np.array( k* [patch_t[:, col],]).transpose()
            cluster = np.argmin(np.sum(temp * temp, axis=0))
            patch_cluster[col] = cluster
        # Update center after each iterations
        center = np.zeros((patch_t.shape[0], k))
        for col_new in range(num):
            cluster_ind = patch_cluster[col_new]
            center[:, int(cluster_ind):int(cluster_ind)+1] += patch_t[:,int(col_new):int(col_new)+1]
            counts[int(cluster_ind)] += 1
        center = center / counts
        # END_YOUR_CODE
    return center

############################################################
# Problem 4.2

def extractFeatures(patches,centroids):
    """
    Given patches for an image and a set of centroids, extracts and return
    the features for that image.
    
    Args:
      patches - 2D numpy array of size patchSize x numPatches
      centroids - 2D numpy array of size patchSize x k
      
    Returns:
      features - 2D numpy array with new feature values for each patch
                 of the image in rows, size is numPatches x k
    """
    k = centroids.shape[1]
    numPatches = patches.shape[1]
    features = np.empty((numPatches,k))
    # BEGIN_YOUR_CODE (around 9 lines of code expected)
    for p in range(numPatches):
        dif = centroids - np.array([patches[:, p], ] * k).transpose()
        total = np.sqrt(np.sum(dif * dif, axis=0))
        j = np.sum(total)/k
        np.array([j, ] * k).transpose()
        features[p, :] = np.array([j, ] * k).transpose() - total
        features_ab = abs(features)
        features = (features + features_ab) / 2
        # END_YOUR_CODE
    return features

############################################################
# Problem 4.3.1

import math
def logisticGradient(theta,featureVector,y):
    """
    Calculates and returns gradient of the logistic loss function with
    respect to parameter vector theta.

    Args:
      theta - 1D numpy array of parameters
      featureVector - 1D numpy array of features for training example
      y - label in {0,1} for training example

    Returns:
      1D numpy array of gradient of logistic loss w.r.t. to theta
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    k = -(2 * y - 1)
    # Calculate the numerator as a
    a = k * math.exp(k * np.sum(featureVector * theta))
    # Calculate the denominator as b
    b = 1 + math.exp(np.sum(featureVector * theta) * k)
    return featureVector * (a / b)
    # raise Exception("Not yet implemented.")
    # END_YOUR_CODE

############################################################
# Problem 4.3.2
    
def hingeLossGradient(theta,featureVector,y):
    """
    Calculates and returns gradient of hinge loss function with
    respect to parameter vector theta.

    Args:
      theta - 1D numpy array of parameters
      featureVector - 1D numpy array of features for training example
      y - label in {0,1} for training example

    Returns:
      1D numpy array of gradient of hinge loss w.r.t. to theta
    """
    # BEGIN_YOUR_CODE (around 6 lines of code expected)
    k = -(2 * y -1)
    if (1 + np.sum(theta * featureVector) * k) > 0:
        return featureVector * k
    else:
        return featureVector * 0
    #raise Exception("Not yet implemented.")
    # END_YOUR_CODE


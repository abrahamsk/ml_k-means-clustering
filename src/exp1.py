#!/usr/bin/env python
# coding=utf-8

# Machine Learning 445
# Homework 5: K-Means Clustering
# Katie Abrahams, abrahake@pdx.edu
# 3/8/16

from input import *

"""
Write a program to implement K-means clustering using Euclidean distance, and to
evaluate the resulting clustering using sum-squared error, sum-squared separation, and entropy.
"""

"""
Experiment 1: Repeat the following 5 times, with different random number seeds.
Run your clustering program on the training data (optdigits.train) with K = 10,
obtaining 10 final cluster centers. (Remember not to use the class attribute in the clustering!)
Your initial cluster centers should be chosen at random, with each attribute Ai being an integer in the range [0,16].
Stop iterating K-Means when all cluster centers stop changing or if the algorithm is stuck in an oscillation.

Choose the run (out of 5) that yields the smallest sum-squared error (SSE).
• For this best run, in your report give the sum-squared error, sum-squared separation,
and mean entropy of the resulting clustering. (See the class slides for definitions.)
• Now use this clustering to classify the test data, as follows:
– Associate each cluster center with the most frequent class it contains.
If there is a tie for most frequent class, break the tie at random.
– Assign each test instance the class of the closest cluster center. Again, ties are broken at random.
Give the accuracy on the test data as well a confusion matrix.
– Note: It’s possible that a particular class won’t be the most common one
for any cluster, and therefore no test digit will ever get that label.
• Calculate the accuracy on the test data and create a confusion matrix for the results on the test data.
• Visualize the resulting cluster centers. That is, for each of the 10 cluster centers,
use the cluster center’s attributes to draw the corresponding digit on an 8 x 8 grid.
You can do this using any matrix-to-bit-map format – e.g., pgm: http://en.wikipedia.org/wiki/Netpbm_format#PGM_example
"""

#################
# Data structures
#################

features_train, labels_train = load_optdigits_data("optdigits/optdigits.train")
features_test, labels_test = load_optdigits_data("optdigits/optdigits.test")
# 3823 training features, 3823 training labels
# 1797 test features, 1797 test labels
# each of the features_train and features_test instances has 64 features

# K = 10 clusters
# each cluster has x number of instances
# each instance is a vector from the training (or test, in testing) data with 64 attributes
clusters = [i for i in xrange(0, 10)]
centers = [i for i in xrange(0, 10)]

# initial cluster centers should be chosen at random, with each attribute Ai being an integer in the range [0,16]
#
# for i in features_train:
#     print i,"\n"
# convert lists to arrays to compute mean for centroid calculation
# compute new centroid with np.mean
# features_train, labels_train = convert_data_to_arrays(labels_train, labels_train)
# features_test, labels_test = convert_data_to_arrays(labels_test, labels_test)
features_train = np.asarray(features_train)
labels_train = np.asarray(labels_train)
features_test = np.asarray(features_test)
labels_test = np.asarray(labels_test)

centroid = []
print features_train.shape
centroid = np.mean(features_train, axis=0)
print centroid

# visualization
# (i*256)/16 to assign to "buckets" for PGM (PGM uses 1-256 instead of 1-16)

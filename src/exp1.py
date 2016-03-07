#!/usr/bin/env python
# coding=utf-8

# Machine Learning 445
# Homework 5: K-Means Clustering
# Katie Abrahams, abrahake@pdx.edu
# 3/8/16

from input import *
import random, numpy as np, math, sys, timing

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

####################
# Program parameters
####################
k = 10


###########
# Functions
###########
def initialize_centers():
    """
    Initialize random centers
    :return:
    """
    centers = [[random.randint(0, 16) for i in xrange(0, 64)] for j in xrange(0, 10)]
    return centers


def compute_euclidean_distances(instances, centers):
    """
    Calculate distances from feature instances to centers
    Euclidean distance from features to centers
    will use minimum distances from f -> c to build clusters
    :param instances:
    :param centers:
    :return:
    """
    # all distances for all centers
    all_distances = []
    # loop 10 times for k = 10
    for i in xrange(0, 10):
        # distances for a single center
        dist_for_one_center = []
        for j in xrange(len(instances)):
            dist_for_one_center.append(euclidean_dist(instances[j], centers[i]))
        # build list of lists for all distances from instances -> centers
        all_distances.append(dist_for_one_center)
    return all_distances


def euclidean_dist(feature_instance, center):
    """
    L2 (Euclidean) distance
    d(x,y)= sqrt(∑(xi − yi)^2)
    Use for distance calculation
    :param feature_instance: instance_features:
    :param center: cluster center:
    :return:
    """
    dist = 0
    # iterate over all feature instances to get distances from centers
    for i in xrange(len(feature_instance)):
        dist += math.pow((feature_instance[i] - center[i]), 2)
        # np.sum is super slow, use math.pow instead (np sum is optimized for arrays)
        # dist = np.sum((feature_instance[i] - center[i]) ** 2)
    return np.sqrt(dist)


def build_clusters(euclidean_distances):
    """
    Build clusters by finding min distances from instances to centers
    :param euclidean_distances:
    :return:
    """
    clusters = [[] for i in xrange(k)]


#######################################################################
# K-means training algorithm (run 5 times)

# Outer loop: run 5 times, per hw
# Inner loop: go until centroid is not moving or until threshold is met
# (distance between prev centroid and current centroid is .1)
#######################################################################
def k_means_training(features_train, labels_train):
    """
    Run k-means algorithm
    :return:
    """
    # initial cluster centers should be chosen at random, with each attribute Ai being an integer in the range [0,16]
    # for each of the 10 centers, generate a length 64 cluster
    # len 10 with 64 attributes at each of the 10 indices
    # print "K-Means training"
    ##############################
    # run k-means training 5 times
    ##############################
    k_means_increment = 1
    for run in xrange(0, 5):
        text = "\rK-means training run " + str(k_means_increment) + "/" + str(5)
        sys.stdout.write(text)
        k_means_increment += 1

        # get initial random clusters
        centers = initialize_centers()
        # compute Euclidean distances from centers to feature instances
        dists = []
        dists = compute_euclidean_distances(features_train, centers)
        # build clusters using minimum distances, pass in list of distances from instances -> centers
        clusters = []
        clusters = build_clusters(dists)


def k_means_testing(features_test, labels_test):
    """
    :param features_test:
    :param labels_test:
    :return:
    """
    print "K-Means Testing"


###############
# Visualization
###############
# (i*256)/16 to assign to "buckets" for PGM (PGM uses 1-256 instead of 1-16)
# create 8x8 matrix:
# print out 8 attributes for each row


######
# Main
######
def main():
    # data structures
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

    # Run k-means
    k_means_training(features_train, labels_train)
    # k_means_testing(features_test, labels_test)


if __name__ == "__main__":
    main()

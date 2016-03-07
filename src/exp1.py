#!/usr/bin/env python
# coding=utf-8

# Machine Learning 445
# Homework 5: K-Means Clustering
# Katie Abrahams, abrahake@pdx.edu
# 3/8/16

from input import *
from kmeans_func import *
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
        num_instances = len(features_train)
        clusters = build_clusters(dists, num_instances)

        # stopping condition
        continue_k_means = True

        # check for empty clusters and retry (reinitialize centers, get distances, and build clusters)
        # if there are any empty clusters present
        check_empty = False
        check_empty = (check_empty_clusters(clusters))
        count_loop = 0
        while check_empty is True:
            # rebuild centers and clusters
            # get initial random clusters
            centers = initialize_centers()
            # compute Euclidean distances from centers to feature instances
            dists = []
            dists = compute_euclidean_distances(features_train, centers)
            # build clusters using minimum distances, pass in list of distances from instances -> centers
            clusters = []
            num_instances = len(features_train)
            clusters = build_clusters(dists, num_instances)
            check_empty = False
            check_empty = (check_empty_clusters(clusters))
            count_loop += 1
            print "\nRebuilding clusters, time", count_loop

        # if there are no empty clusters, proceed
        if check_empty is False:
            # copy centers to compare with new centers
            store_centers = [row[:] for row in centers]
            # recompute the center of each cluster
            centers = update_centers(clusters, features_train)
            dists = compute_euclidean_distances(features_train, centers)
            clusters = build_clusters(dists, num_instances)



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
    # clusters = [i for i in xrange(0, 10)]
    # centers = [i for i in xrange(0, 10)]

    # Run k-means
    k_means_training(features_train, labels_train)
    # k_means_testing(features_test, labels_test)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding=utf-8

# Machine Learning 445
# Homework 5: K-Means Clustering
# Katie Abrahams, abrahake@pdx.edu
# 3/8/16

from input import *
import random, numpy as np, timing

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


###########
# Functions
###########

def euclidean_dist(x, y):
    """
    L2 (Euclidean) distance
    Use for distance calculation
    :param x: instance_features or clusters, depending on args:
    :param y: center or clusters, depending on args:
    :return:
    """
    return np.sqrt(np.sum(((f - c) ** 2 for f, c in zip(x, y))))
    # use center[1] when passing in centers enumerated, to access centers and not identifying cluster #


def initialize_clusters():
    """
    Initialize random centers and empty centroids
    :return:
    """
    centers = [[random.randint(0, 16) for i in xrange(0, 64)] for j in xrange(0, 10)]
    centroids = [[0 for i in xrange(0, 64)] for j in xrange(0, 10)]
    return centers, centroids


def check_cluster_centers(centers, new_centers):
    """
    Stop iterating K-Means when all cluster centers stop changing
    or if the algorithm is stuck in an oscillation.
    Use d or d^2 (Euclidean distance) to compute distances between each index of centers
    :param centers:
    :param new_centers:
    :return false if cluster centers have stopped moving (or threshold of .1 has been met):
    """
    # check for empty new_clusters (no new centroids during first k-means iteration)
    if len(new_centers) == 0:
        return True

    distances = []
    # for i in xrange(len(clusters)):
    #     for j in clusters[i]:
    #         distances.append((euclidean_dist(clusters[i, j], new_clusters[i, j])))

    # enumerate(thing), where thing is either an iterator or a sequence,
    # returns a iterator that will return (0, thing[0]), (1, thing[1]), (2, thing[2]), etc
    # for i, j in enumerate(centers):
    #     for k, l in enumerate(centers[i]):
    #         distances.append(euclidean_dist(centers[i][k], new_centers[i][k]))
    for c in centers:
        for nc in new_centers:
            # find minimum distance
            distances.append(euclidean_dist(c, nc))

    met_threshold = []
    for d in distances:
        if (d <= 0.1):
            met_threshold.append(d)

    # if all clusters have stopped moving
    if len(met_threshold) == len(distances):
        return False
    # if clusters are still moving, keep going
    else:
        return True


def sum_squared_error(feature_instances, centroids):
    """
    Compute sum squared error from feature instances to center of clusters
    :param features_train:
    :param centroids:
    :return:
    """



#######################################################################
# Training (run 5 times)

# Outer loop: run 5 times, per hw
# Inner loop: go until centroid is not moving or until threshold is met
# (distance between prev centroid and current centroid is .1)
#######################################################################
def k_means(features_train, labels_train, features_test, labels_test):
    """
    Run k-meas algorithm
    :return:
    """
    # initial cluster centers should be chosen at random, with each attribute Ai being an integer in the range [0,16]
    # for each of the 10 centers, generate a length 64 cluster
    # len 10 with 64 attributes at each of the 10 indices
    centers, centroids = initialize_clusters()

    # Stop iterating K-Means when all cluster centers stop changing
    # or if the algorithm is stuck in an oscillation.
    do_continue = check_cluster_centers(centers, centroids)
    while (do_continue):
        ################################################
        # Form K clusters by assigning each point to its
        # closest centroid using Euclidean distance.
        # Find closest center for each instance
        # by iterating through centers and finding min
        # distance from instance to one of the centers.
        ################################################
        # K clusters to match K centers
        # with nested lists for centers and a list of feature instances
        clusters = [[] for i in xrange(len(centers))]
        # minimum distances between feature vectors and centers
        min_dist = []
        # distances to compare to get minimum
        compare_dist = []
        # keep track of center with min distance
        track_center = []
        # center with min distance to feature vector
        min_dist_center = []

        # distances from instances to centers
        dists = []
        # Euclidean distance from features to centers
        # find minimum distances from f -> c and build clusters
        # out of features with min distance to center
        for f in features_train:
            for c in centers:
                # find minimum distance
                dists.append(euclidean_dist(f, c))
            idx = dists.index(min(dists))
            clusters[idx].append(f)
            # clear distances list
            dists = []

        ########################################
        # Recompute the centroid of each cluster
        ########################################
        # convert lists to arrays to compute mean for centroid calculation
        # compute new centroid with np.mean
        features_train = np.asarray(features_train)
        labels_train = np.asarray(labels_train)
        features_test = np.asarray(features_test)
        labels_test = np.asarray(labels_test)
        # features_train.shape = (3823, 64)
        # random list so length of list is correct
        centroids = [[0 for i in xrange(0, 64)] for j in xrange(0, 10)]
        # for each cluster, take mean of of the instances in the cluster to get new centroid

        # If a cluster is empty:
        # Choose a replacement centroid
        # at random, or from cluster that has highest SSE
        for i in clusters:
            if len(i) == 0:  # possible to have zero-length clusters
                print "Empty cluster"

        for i in xrange(len(centers)):
            # print clusters[i]
            # print centroids[i]  # all zeroes
            centroids[i] = np.mean(clusters[i], axis=0)  # np.mean(cluster, axis = 0)
        # print centroids

        # compute SSE
        sum_sq_error = sum_squared_error(features_train, centroids)


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
    ############# for l in xrange(0, 5):
    k_means(features_train, labels_train, features_test, labels_test)


if __name__ == "__main__":
    main()

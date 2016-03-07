#!/usr/bin/env python
# coding=utf-8

# Machine Learning 445
# Homework 5: K-Means Clustering
# Katie Abrahams, abrahake@pdx.edu
# 3/8/16

from input import *
import random, numpy as np, math, sys, timing

####################
# Program parameters
####################
k = 10
# number of training instances
num_training_instances = 3823


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


def build_clusters(euclidean_distances, num_instances):
    """
    Build clusters by finding min distances from instances to centers
    :param euclidean_distances:
    :param num_instances:
    :return:
    """
    clusters = [[] for i in xrange(k)]

    # iterate through the number of instances
    for i in xrange(len(euclidean_distances[0])):
        dist_temp_collection = []
        # iterate though k times
        for j in xrange(k):
            dist_temp_collection.append(euclidean_distances[j][i])

        # find minimum distances
        # min_from_dist = min(dist_temp_collection)
        min_val, min_val_idx = min((min_val, min_val_idx) for (min_val_idx, min_val) in enumerate(dist_temp_collection))

        # build clusters using min distances
        # save indexes for instances with min distances
        clusters[min_val_idx].append(i)

    # print "\n", len(clusters[0])
    # print clusters
    return clusters


def check_empty_clusters(clusters):
    """
    Check if there are empty clusters and rebuild if so
    :param clusters:
    :return:
    """
    empty_clusters = False
    for c in xrange(len(clusters)):
        if len(clusters[c]) == 0:
            empty_clusters = True
    return empty_clusters


def update_centers(clusters, instances):
    """
    Recompute centroid of each cluster
    :param clusters:
    :param instances:
    :return:
    """
    new_centers = []
    # iterate k times
    for i in xrange(k):
        one_cluster = get_features_for_cluster(clusters[i], instances)
        new_centers.append(np.mean(np.asarray(one_cluster), axis=0).tolist())
    return new_centers


def get_features_for_cluster(cluster, instances):
    """
    Get associated features for a cluster
    :param cluster:
    :param instances:
    :return:
    """
    features = []
    for c in xrange(len(cluster)):
        features.append(instances[cluster[c]])
    # print features
    return features


def check_stopping_cond(store_centers, centers):
    """
    Check loop stopping condition while running k-means
    by comparing distances between current (stored) clusters and new clusters
    :param store_centers:
    :param centers:
    :return:
    """
    check_threshold_stopping_cond = False
    dists = []
    for i in xrange(k):
        dists.append(euclidean_dist(store_centers[i], centers[i]))
        # print "dists", dists
        for d in dists:
            if d <= .001:
                check_threshold_stopping_cond = True
            else:
                check_threshold_stopping_cond = False
    return check_threshold_stopping_cond

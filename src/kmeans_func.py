#!/usr/bin/env python
# coding=utf-8

# Machine Learning 445
# Homework 5: K-Means Clustering
# Katie Abrahams, abrahake@pdx.edu
# 3/8/16

from __future__ import division
import random, numpy as np, math

##################################
# Functions for k-means clustering
# used in experiments 1 and 2
##################################

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


def sum_squared_error(clusters, centers, instances):
    """
    Calculate sum squared error (SSE)
    SSE=∑       ,∑   d(x,mi)^2
        i=1 to K,x∈Ci
    where Ci is the ith cluster, mi is the centroid of cluster Ci
    (the mean vector of all the data points in Ci), and
    d(x, mi) is the distance between data point x and centroid mi.
    We want to minimize internal coherence of each cluster – i.e., minimize SSE.
    :param clusters:
    :param centers:
    :param instances:
    :return:
    """
    sse = 0
    # use Euclidean distance to get SSE
    # iterate k times (for k centers)
    for l in xrange(k):
        for m in xrange(len(clusters[l])):
            n = clusters[l][m]
            # square result of euclidean distance for sse
            sse += (euclidean_dist(instances[n], centers[l])) ** 2
    return sse


def sum_squared_separation(centers):
    """
    Calculate sum squared separation (SSS)
    Sum Squared Separation (clustering)
    = ∑ d(mi, mj)^2
      (all distinct pairs of clusters i, j (i≠j))
    We want to maximize pairwise separation of each cluster – i.e., maximize SSS
    :param centers
    :return:
    """
    sum_sq_separation = 0
    for d in xrange(k - 1):
        c = d + 1
        while c < k:
            # run euclidean distance from different clusters to get SSS
            # square result of euclidean distance for sss
            sum_sq_separation += (euclidean_dist(centers[d], centers[c])) ** 2
            c += 1
    return sum_sq_separation


def entropy(cluster, instances_labels):
    """
    Find entropy of a single cluster
    Entropy of a cluster: The degree to which a cluster consists of objects of a single class.
                  |classes|
    entropy(Ci) = − ∑        pi,j log2 pi,j
                   j=1
    where
    pi, j = probability that a member of cluster i belongs to class j
    = mi,j/mi , where mi,j is the number of instances in cluster i with class j
    and mi is the number of instances in cluster i
    :param cluster: one cluster:
    :param instances_labels: real value of instances:
    :return ent:
    """
    # entropy for cluster passed in arguments
    ent = 0
    # count mi,j for numerator (number of instances in cluster i with class j)
    count_instances_in_cluster = [0 for i in xrange(k)]
    for c in xrange(len(cluster)):
        i = cluster[c]
        j = instances_labels[i]
        count_instances_in_cluster[j] += 1

    # count classes per cluster
    # iterator for the number of classes (= k = 10)
    for i in xrange(k):
        # numerator m i,j is the number of instances in cluster i with class j
        numerator = count_instances_in_cluster[i]
        # denominator mi is the number of instances in cluster i
        denominator = len(cluster)
        # avoid math domain errors by catching cases where num instances in cluster is 0
        if count_instances_in_cluster[i] == 0:
            ent = 0
        else:
            # sum for |classes| times to get entropy for a single cluster
            ent += (numerator / denominator) * math.log((numerator / denominator), 2)

    # negate entropy (account for negative in entropy eq)
    return -ent


def mean_entropy(clusters, instances_labels):
    """
    Find mean entropy of a clustering
    We want to minimize mean entropy
    Mean entropy of a clustering: Average entropy over all clusters in the clustering
                                K
    mean entropy (Clustering) = ∑ mi/m entropy(Ci)
                                1
    where mi (numerator) is the number of instances in cluster i
    and m (denominator) is the total number of instances in the clustering.
    :param clusters: all clusters:
    :param instances_labels: real value of instances:
    :return mean_ent:
    """
    mean_ent = 0
    # iterate for the total number of clusters/k times
    for c in xrange(len(clusters)):
        # numerator for multiplier of individual cluster entropy is len of one cluster
        numerator = len(clusters[c])
        # denominator for multiplier of individual cluster entropy is the total number of instances
        denominator = len(instances_labels)
        # sum entropy for all clusters multiplied by mi/m to get mean entropy
        mean_ent += (numerator / denominator) * entropy(clusters[c], instances_labels)

    return mean_ent


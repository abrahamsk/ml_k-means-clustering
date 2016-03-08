#!/usr/bin/env python
# coding=utf-8

# Machine Learning 445
# Homework 5: K-Means Clustering
# Katie Abrahams, abrahake@pdx.edu
# 3/8/16

from __future__ import division
import random, numpy as np, math, array

##################################
# Functions for k-means clustering
# used in experiments 1 and 2
##################################

####################
# Program parameters
####################
# !! only use the k corresponding to the experiment you want to run
# and comment out the other k value

# exp 1:
# k = 10
# exp 2:
k = 30

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
    centers = [[random.randint(0, 16) for i in xrange(0, 64)] for j in xrange(0, k)]
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
    # loop k times for k = 10 or 30 (exp1 or 2)
    for i in xrange(0, k):
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
        # save indexes for instances with min distances to centers
        clusters[min_val_idx].append(i)

    return clusters


def check_empty_clusters(clusters):
    """
    Check if there are empty clusters in the set of all clusters
    :param clusters:
    :return:
    """
    empty_clusters = False
    for c in xrange(len(clusters)):
        if len(clusters[c]) == 0:
            empty_clusters = True
    return empty_clusters


def recompute_centroids(clusters, instances):
    """
    Recompute centroid of each cluster
    :param clusters:
    :param instances:
    :return:
    """
    new_centroids = []
    # iterate k times ( = num centers)
    for i in xrange(k):
        one_cluster = get_features_for_cluster(clusters[i], instances)
        new_centroids.append(np.mean(np.asarray(one_cluster), axis=0).tolist())
    return new_centroids


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
        # set up indices
        c = d + 1
        # loop while c is less than num centers
        while c < k:
            # run euclidean distance from different clusters to get SSS
            # square result of euclidean distance for SSS
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
    # iterator for the number of classes (= k)
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


###############################
# Functions for k-means testing
###############################
def get_most_freq_classes(clusters, labels):
    """
    # - Associate each cluster center with the most frequent class it contains.
    # If there is a tie for most frequent class, break the tie at random
    # - Assign each test instance the class of the closest cluster center.
    :param clusters:
    :param labels:
    :return:
    """
    most_frequent_class = []
    # iterate through clusters and find most frequent class for each cluster
    for i in xrange(len(clusters)):
        most_frequent_class.append(most_freq_class(clusters[i], labels))
    return most_frequent_class


def most_freq_class(cluster, label):
    """
    Get most frequent class of in a cluster
    # Associate each cluster center with the most frequent class it contains.
    # If there is a tie for most frequent class, break the tie at random
    :param cluster:
    :param label:
    :return:
    """
    count_class = [0 for i in xrange(k)]
    # loop for all items in a cluster and count labels in cluster
    for i in xrange(len(cluster)):
        count_class[label[cluster[i]]] += 1
    # print "count class:", count_class
    # find the most frequent class
    max_class_val = max(count_class)
    max_index = count_class.index(max_class_val)
    # If there is a tie for most frequent class, break the tie at random
    break_tie = []
    for i in xrange(k):
        if max_class_val == count_class[i]:
            break_tie.append(i)
    # print "break tie:", break_tie
    if break_tie:
        choice = random.choice(break_tie)
        return choice

    return max_index


def confusion_matrix(most_freq_classes, test_clusters, labels_test):
    """
    Create a confusion matrix for the results on the test data
    Call accuracy function for accuracy of test data
    :param most_freq_classes:
    :param test_clusters:
    :param labels_test:
    :return:
    """
    # num correct
    correct = 0
    # num total
    total = 0
    # test accuracy
    acc = 0.0
    # col headers (0-9)
    predicted = [i for i in xrange(k)]
    # row headers (0-9)
    actual = [i for i in xrange(k)]
    # confusion matrix
    conf_matrix = [[0 for i in xrange(k)] for i in xrange(k)]

    # build confusion matrix by iterating through all test clusters and comparing class to actual class
    # all clusters
    for i in xrange(len(test_clusters)):
        # individual cluster
        for j in xrange(len(test_clusters[i])):
            # get actual and predicted class to compare
            cluster_class = test_clusters[i][j]
            actual_class = labels_test[cluster_class]
            predicted_class = most_freq_classes[i]
            # append to confusion matrix
            conf_matrix[predicted_class][actual_class] += 1
            # if predicted matches actual, increment correct count
            if predicted_class == actual_class:
                correct += 1
            # increment total whether or not prediction is correct
            total += 1

    # output confusion matrix
    print "---------------"
    print "         Confusion matrix"
    print "         Predicted Class"
    print " ", predicted
    for i in xrange(len(conf_matrix)):
        print actual[i], conf_matrix[i]

    print "---------------"
    acc = test_accuracy(correct, total)
    print "---------------"
    return acc


def test_accuracy(correct_results, total_results):
    """
    Calculate the accuracy on the test data
    using the confusion matrix computations
    :param correct_results:
    :param total_results:
    :return:
    """
    accuracy = correct_results / total_results
    print "Test accuracy:", accuracy
    return accuracy


#############################
# Visualization for test data
#############################
def visualization_results(center, idx, exp_num):
    """
    Visualize the resulting cluster centers.
    For each of the k (10 or 30) cluster centers, use the cluster center’s attributes
    to draw the corresponding digit on an 8 x 8 grid.
    ref: en.wikipedia.org/wiki/Netpbm_format#PGM_example
    :param center:
    :param idx:
    :param exp_num:
    :return:
    """
    # create 8x8 matrix: print out 8 attributes for each row
    width = 8
    height = 8

    # initialize with random values
    arr = array.array('B')

    # write center values to array
    for i in xrange(0, width * height):
        arr.append(int(round(center[i])) * 16)

    # save to file
    save_as = exp_num + "_" + str(idx) + ".pgm"
    fout = open("pgm/" + save_as, 'wb')

    # PGM Header
    header = 'P5' + '\n' + str(width) + '  ' + str(height) + '  ' + str(255) + '\n'

    # write data to file and close
    fout.write(header)
    arr.tofile(fout)
    fout.close()

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
Experiment 2: Run K-means on the same data but with K = 30
"""

####################
# Program parameters
####################
exp_num = "exp2"


#######################################################################
# K-means training algorithm (run 5 times)

# Outer loop: run 5 times, per hw
# Inner loop: go until centroid is not moving or until threshold is met
# (distance between prev centroid and current centroid is .1)
#######################################################################
def k_means_training(features_train, labels_train):
    """
    Run k-means algorithm
    :param features_train:
    :param labels_train:
    :return best_sse:
    :return best_centers:
    """
    # initial cluster centers should be chosen at random, with each attribute Ai being an integer in the range [0,16]
    # for each of the 30 centers, generate a length 64 cluster
    # len 30 with 64 attributes at each of the 30 indices
    print "K-Means training..."

    # track SSE over k-means runs
    sum_sq_errors = []
    # store centers/clusters and save those from best SSE run
    best_centers = []
    best_clusters = []
    # track num k-means runs
    k_means_increment = 1
    ##############################
    # run k-means training 5 times
    ##############################
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

        # check for empty clusters and retry (reinitialize centers, get distances, and build clusters)
        # if there are any empty clusters present
        check_empty = False
        check_empty = (check_empty_clusters(clusters))
        # keep a count of the num times there are empty clusters and rebuild if needed
        recluster_count = 0
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
            # check for empty clusters in rebuilt clusters
            check_empty = False
            check_empty = (check_empty_clusters(clusters))
            recluster_count += 1
            print "\nRebuilding clusters, time", recluster_count

        # track how many times kmeans runs
        k_means_iter_counter = 0
        # to store current centers for comparison
        store_centers = []
        # if there are no empty clusters, proceed
        if check_empty is False:
            # continue until stopping conditions are met:
            # stop iterating K-Means when all cluster centers stop changing
            # or if the algorithm is stuck in an oscillation
            while store_centers != centers:
                k_means_iter_counter += 1
                print "Old and new centers still not equal..."
                # copy centers to compare with new centers
                store_centers = [row[:] for row in centers]
                # recompute the center of each cluster
                centers = recompute_centroids(clusters, features_train)
                dists = compute_euclidean_distances(features_train, centers)
                clusters = build_clusters(dists, num_instances)
                check_empty = (check_empty_clusters(clusters))
                if check_empty:
                    print "New cluster is empty"
        print "\nK-means ran", k_means_iter_counter, "time(s)"
        # print "Dist between old and new clusters", cluster_dist

        # compute sum squared error to select best run out of 5
        current_sse = sum_squared_error(clusters, centers, features_train)
        # save initial values for best centers/clusters
        best_centers = [row[:] for row in centers]
        best_clusters = [row[:] for row in clusters]
        # if list has items in it, compared to current SSE and save clusters if current SSE is the best
        if sum_sq_errors:
            # get best stored SSE
            min_sse_from_list, min_sse_idx = min(
                (min_sse_from_list, min_sse_idx) for (min_sse_idx, min_sse_from_list) in enumerate(sum_sq_errors))
            # compare current SSE to all the rest in the list
            # if this is the best SSE yet, save current centers as the best
            if current_sse < min_sse_from_list:
                best_centers = [row[:] for row in centers]
                # also save clusters for computing mean entropy
                best_clusters = [row[:] for row in clusters]
        # add to list of all SSEs
        sum_sq_errors.append(current_sse)

    # find min SSE
    min_val, min_val_idx = min((min_val, min_val_idx) for (min_val_idx, min_val) in enumerate(sum_sq_errors))
    print "Min SSE:", min_val
    # print "All SSE", sum_sq_errors

    # Choose the run (out of 5) that yields the smallest sum-squared error (SSE)
    # For this best run, in your report give the sum-squared error,
    # sum-squared separation, and mean entropy of the resulting clustering.
    # See k_means_training_stats

    # only need to keep centers from best k-means run for testing
    with open('centers_outfile_exp2', 'w') as file_centers:
        file_centers.writelines('\t'.join(str(j) for j in i) + '\n' for i in best_centers)

    # keep clusters from best centers for computing mean entropy
    with open('clusters_outfile_exp2', 'w') as file_clusters:
        file_clusters.writelines('\t'.join(str(j) for j in i) + '\n' for i in best_clusters)

    # return best SSE, best centers
    return min_val, best_centers


def k_means_training_stats(labels_train):
    """
    Calculate sum squared separation and mean entropy
    for the best run out of the five training runs
    Run k_means_training before this function
    or the outfile with cluster centers won't be populated
    :param labels_train:
    :return sum_squared_separation, mean_entropy:
    """
    # read in centers from best of 5 k-means training runs
    centers = open("centers_outfile_exp2", "r")
    best_centers = []
    for line in centers:
        # Split the line on runs of whitespace
        number_strings = line.split()
        numbers = [n for n in number_strings]
        numbers_float = [float(i) for i in numbers]
        # Add the row to the list
        best_centers.append(numbers_float)

    # read in clusters from best of 5 k-means training runs
    clusters = open("clusters_outfile_exp2", "r")
    best_clusters = []
    for line in clusters:
        # Split the line on runs of whitespace
        number_strings = line.split()
        numbers = [n for n in number_strings]
        numbers_float = [int(i) for i in numbers]
        # Add the row to the list
        best_clusters.append(numbers_float)

    # sum squared separation (best run)
    print "Getting sum squared separation for best clusters..."
    sum_squared_sep = sum_squared_separation(best_centers)

    # mean entropy (best run)
    print "Getting mean entropy..."
    mean_ent = mean_entropy(best_clusters, labels_train)

    return sum_squared_sep, mean_ent


def k_means_testing(features_test, labels_test):
    """
    Run k_means_training before this function
    or the outfile with cluster centers won't be populated

    Use training clustering to classify the test data, as follows:
    1. Associate each cluster center with the most frequent class it contains.
        If there is a tie for most frequent class, break the tie at random.
    2. Assign each test instance the class of the closest cluster center.
        Again, ties are broken at random. Give the accuracy on the test data as well a confusion matrix.
    - Note: It’s possible that a particular class won’t be the most common one
        for any cluster, and therefore no test digit will ever get that label.
    3. Calculate the accuracy on the test data and create a confusion matrix for the results on the test data.
    4. Visualize the resulting cluster centers.
        For each of the 30 cluster centers, use the cluster center’s attributes
        to draw the corresponding digit on an 8 x 8 grid.
    :param features_test:
    :param labels_test:
    :return:
    """
    print "K-Means testing..."
    # read in centers from best of 5 k-means training runs
    centers = open("centers_outfile_exp2", "r")
    best_centers = []
    for line in centers:
        # Split the line on runs of whitespace
        number_strings = line.split()
        numbers = [n for n in number_strings]
        numbers_float = [float(i) for i in numbers]
        # Add the row to the list
        best_centers.append(numbers_float)

    # compute Euclidean distances from centers to feature instances for test data
    # use best centers from training data
    test_dists = []
    test_dists = compute_euclidean_distances(features_test, best_centers)

    # build test clusters using minimum distances, pass in list of distances from instances -> centers
    test_clusters = []
    num_instances = len(features_test)
    test_clusters = build_clusters(test_dists, num_instances)

    # check for empty clusters
    check_empty = (check_empty_clusters(test_clusters))
    # check that there are no empty clusters
    if check_empty is False:
        print "No empty test clusters"

    # 1. Associate each cluster center with the most frequent class it contains.
    # If there is a tie for most frequent class, break the tie at random
    # 2. Assign each test instance the class of the closest cluster center.
    most_freq_classes = []
    most_freq_classes = get_most_freq_classes(test_clusters, labels_test)
    print "Most frequent classes:", most_freq_classes

    # 3. Calculate the accuracy on the test data and create a confusion matrix for the results on the test data.
    # accuracy is calculated in confusion matrix function
    test_acc = confusion_matrix(most_freq_classes, test_clusters, labels_test)

    # 4. Visualize the resulting cluster centers.
    # For each of the 30 cluster centers, use the cluster center’s attributes
    # to draw the corresponding digit on an 8 x 8 grid.
    for i in xrange(len(best_centers)):
        visualization_results(best_centers[i], i, exp_num)


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

    # K = 20 clusters
    # each cluster has x number of instances
    # each instance is a vector from the training (or test, in testing) data with 64 attributes
    # clusters = [i for i in xrange(0, 30)]
    # centers = [i for i in xrange(0, 30)]

    # Run k-means
    # run training to get SSE, then comment out to save runtime
    sum_sq_error, best_centers = k_means_training(features_train, labels_train)
    print "K-means training complete"
    print "-------------------------"
    print "Sum squared error from the best of 5 runs:", sum_sq_error

    # get sum squared separation and mean entropy
    sum_sq_sep, mean_ent = k_means_training_stats(labels_train)
    print "Sum squared separation:", sum_sq_sep
    print "Mean entropy:", mean_ent

    print "-------------------------"

    k_means_testing(features_test, labels_test)
    print "K-means testing complete"
    print "-------------------------"


if __name__ == "__main__":
    main()

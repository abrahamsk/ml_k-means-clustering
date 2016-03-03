#!/usr/bin/env python
# coding=utf-8

# Machine Learning 445
# Homework 5: K-Means Clustering
# Katie Abrahams, abrahake@pdx.edu
# 3/8/16


import numpy as np


# use modified TA-provided code from HW3 for data input
def load_optdigits_data(filename):
    """
    Each line in the datafile is a csv with features values, followed by a label, per sample; one sample per line
    """

    "The file function reads the filename from the current directory, unless you provide an absolute path " \
    "e.g. /path/to/file/file.py or C:\\path\\to\\file.py"

    unprocessed_data_file = file(filename, 'r')

    "Obtain all lines in the file as a list of strings."

    unprocessed_data = unprocessed_data_file.readlines()

    labels = []
    features = []

    for line in unprocessed_data:
        feature_vector = []

        "Convert the String into a list of strings, being the elements of the string separated by commas"
        split_line = line.split(',')

        "Iterate across elements in the split_line except for the final element "
        for element in split_line[:-1]:
            feature_vector.append(float(element))

        "Add the new vector of feature values for the sample to the features list"
        features.append(feature_vector)

        "Obtain the label for the sample and add it to the labels list"
        labels.append(int(split_line[-1]))

    "Return List of features with its list of corresponding labels"
    return features, labels


def convert_data_to_arrays(features, labels):
    """
    conversion to a numpy array is easy if you're starting with a List of lists.
    The returned array has dimensions (M,N), where M is the number of lists and N is the number of

    """

    return np.asarray(features), np.asarray(labels)


def main():
    features_train, labels_train = load_optdigits_data("optdigits/optdigits.train")
    features_test, labels_test = load_optdigits_data("optdigits/optdigits.test")

    features_train, labels_train = convert_data_to_arrays(labels_train, labels_train)
    features_test, labels_test = convert_data_to_arrays(labels_test, labels_test)


if __name__ == "__main__":
    main()

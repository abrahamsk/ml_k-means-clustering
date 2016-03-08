## Running instructions:

#### Option 1: 
Generate your own centers, clusters, and sum squared error

*Ensure k-means training is NOT commented out:*
```
sum_sq_error, best_centers = k_means_training(features_train, labels_train)
```

#### Option 2: 
Use included centers, clusters, and sum squared error from best of 5 k-means training runs

*Comment out k-means training:*
```
# sum_sq_error, best_centers = k_means_training(features_train, labels_train)
```

*Make sure `centers_outfile` and `clusters_outfile` are at the same file level as exp1 and exp2*

#### Dependencies:
The Python dependencies included at the top of each file must be installed in user's Python env

### With either Option 1 or Option 2 complete and dependencies installed:
In your IDE of choice or in the Python command line:

1. Run Experiment 1 by executing `exp1.py`

2. Run Experiment 2 by executing `exp2.py`

#### Etc:
Built with Python 2.7, Python 3.0 support is untested

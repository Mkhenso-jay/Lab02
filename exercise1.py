# Exercise 1: Prepare dataset for perceptron
# This script defines the training dataset X and labels y, shuffles the dataset,
# and prepares it for training the perceptron.

import numpy as np

# Training Dataset
X = np.array([
    [1, 5], [2, 8], [1, 10], [2, 12], [3, 15],
    [1, 7], [2, 9], [3, 11], [4, 12], [1, 6],
    [2, 10], [3, 14], [1, 8], [2, 13], [3, 15],
    [4, 18], [1, 7], [2, 9], [3, 12], [4, 14],
    [1, 6], [2, 8], [3, 10], [4, 11], [2, 7],
    [6, 12], [7, 29], [8, 64], [6, 15], [7, 70],
    [8, 59], [9, 77], [10, 17], [7, 100], [6, 5],
    [8, 91], [9, 47], [10, 66], [7, 87], [6, 41],
    [8, 25], [9, 63], [10, 95], [7, 21], [6, 44],
    [8, 83], [9, 37], [10, 99], [7, 70], [6, 58]
])

y = np.array([
    # Not Popular
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,
    # Popular
    1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1
])

# Shuffle dataset
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

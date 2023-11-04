import numpy as np
import random

"""Python script which contains all the helper functions for the preprocessing"""

def normalize(arr, t_min, t_max):
    """Implementation of Min-Max Normalization

    Args:
        arr (np.array): data
        t_min (int): lower boundary for normalization
        t_max (int): upper boundary for normalization

    Returns:
        array: normalized values
    """
    norm_arr = []
    diff = t_max - t_min
    diff_arr = arr.max() - arr.min() 
    for i in arr:
        temp = (((i - arr.min()))*diff)/diff_arr + t_min
        norm_arr.append(temp)
    return np.array(norm_arr)

def encode_targets(arr):
    """One-Hot Encoding of target values

    Args:
        arr (np.array): Array containing target values

    Returns:
        array: encoded targets
    """
    enc_targets = np.zeros((arr.size, arr.max() + 1))
    enc_targets[np.arange(arr.size), arr] = 1
    return enc_targets

def batch_generator(inputs, targets, batch_size):
    """Shuffles data and creates minibatches

    Args:
        inputs (array): Input data
        targets (array): Targets
        batch_size (int): size of each minibatch

    Yields:
        array: one batch of the shuffled data
        array: one batch of the shuffled targets
    """
    combined = list(zip(inputs, targets))  # Combine inputs and targets into pairs
    random.shuffle(combined)  # Shuffle the pairs

    #iterate with step_size = minibatch_size to create batches
    for i in range(0, len(combined), batch_size):
        #slice into arrays with len = minibatch_size
        batch = combined[i: i + batch_size]
        data_batch, target_batch = zip(*batch)
        yield np.array(data_batch), np.array(target_batch)
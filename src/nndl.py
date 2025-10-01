"""nndl module

This module contains functions that are shown in the Chollet
"Neural Networks and Deep Learning" course textbook, and that are
generally useful and reused in various places for the
lecture notebooks.
"""
import numpy as np
import matplotlib.pyplot as plt


def vectorize_samples(samples, dimension=10000):
    """Vectorize word index sequences into a multi-hot encoding.
    The sequences is an array of regular python lists.  We hot encode
    each word we find in the list into a vector of the indicated number
    of feature dimensions, one vector for each of the word sequences /
    reviews.

    Arguments
    ---------
    samples - A numpy array of shape (samples,).  We expect each item in the array
      to be an object type, which will be a Python list or equivalent sequence of
      word indexes
    dimension - Default to 10000 for this problem, but if we load the imdb with
      a different max words we would need to adjust.  Though we could infer this
      dynamically but examing the samples as we did before if we wanted to.

    Returns
    -------
    multi_hot_encoded_samples - A tensor of shape (samples, dimension) where each sample
       has multi-hot encoded the original sequential word list review.
    """
    # initialize a tensor/array of the correct shape, initialized to all 0's
    multi_hot_encoded_samples = np.zeros((len(samples), dimension))
    
    # set specific index to 1 for each word index encountered in the review
    for sample_num, sample in enumerate(samples):
        for word_index in sample:
            # we could increment the count here if wanted a multi-hot encoding
            # counting the number of times the word occurs
            multi_hot_encoded_samples[sample_num, word_index] = 1
            
    return multi_hot_encoded_samples


def plot_history(ax, history_dict, metric_key):
    """Plot the asked for metrics. Usually we need to plot the metric from the training
    data and its corresponding measurement using the validation data, thus we pass in
    two keys for the training and validation metric to plot.

    Arguments
    ---------
    ax - a matplotlib figure axis to create plot onto
    history_dict - A Python dictionary whose keys should return list like enumerable
      items holding the measured metrics over some number of epochs of training.
    metric_key - The string key for the metric, validation data is assumed to be
      accessible as "val_" + metric_key


    """
    # setup epochs and keys/labels for the plot
    train_key = metric_key
    train_label = "Training " + metric_key
    val_key = "val_" + metric_key
    val_label = "Validation " + metric_key
    epochs = np.arange(1, len(history_dict[train_key]) + 1)
    
    # create the plot of the train and test metric
    ax.plot(epochs, history_dict[train_key], 'r-', label=train_label)
    ax.plot(epochs, history_dict[val_key], 'b-', label=val_label)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric_key)
    #ax.set_xticks(epochs)
    ax.grid()
    ax.legend();
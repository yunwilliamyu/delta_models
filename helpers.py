#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

def delta_weights(weights1, weights2):
    """Generates ans = weights2 - weights1"""
    weights_diff = []
    assert(len(weights1) == len(weights2)) # otherwise cannot subtract
    for w1, w2 in zip(weights1, weights2):
        if type(w1) is np.ndarray:
            assert(type(w2) is np.ndarray)
            w3 = w2 - w1
            weights_diff.append(w3)
        elif type(w1) is np.bool_:
            assert(type(w2) is np.bool_)
            assert(w1 == w2)
            weights_diff.append(w1)
    return weights_diff

def sparsify_weights(weights, threshold=0.07):
    weights_copy = np.copy(weights)
    for w in weights_copy:
        if type(w) is np.ndarray:
            w[np.abs(w)<threshold]=0
        elif type(w) is np.bool_:
            pass
    return weights_copy


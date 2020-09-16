#!/usr/bin/env python3

# https://towardsdatascience.com/simple-bert-using-tensorflow-2-0-132cb19e9b22

import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import bert
FullTokenizer = bert.bert_tokenization.FullTokenizer
from tensorflow.keras.models import Model       # Keras is the new high level API for TensorFlow
import math
import helpers

max_seq_length = 128  # Your choice here.
input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=True)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])

weights = model.get_weights()

sw = helpers.sparsify_weights(weights)
np.savez('uncompressed_weights.npz', weights)
np.savez_compressed('compressed_weights.npz', weights)
np.savez_compressed('sparse_compressed_weights.npz', sw)

import tensorflow as tf
import numpy as np

def get_tf_dataset(inputs, batch_size, train=True):
    if len(inputs) == 3:
        x, s, y = inputs
        n = x.shape[0]

        data = tf.data.Dataset.from_tensor_slices(x.astype(np.float32))
        labels = tf.data.Dataset.from_tensor_slices(y.astype(np.float32))
        mask = tf.data.Dataset.from_tensor_slices(s.astype(np.float32))

        dataset = (tf.data.Dataset.zip((data, mask, labels))
                   .shuffle(n).batch(batch_size).prefetch(4))
        return dataset

    elif len(inputs) == 2:
        x, s = inputs
        n = x.shape[0]

        data = tf.data.Dataset.from_tensor_slices(x.astype(np.float32))
        mask = tf.data.Dataset.from_tensor_slices(s.astype(np.float32))

        dataset = (tf.data.Dataset.zip((data, mask))
                   .shuffle(n).batch(batch_size).prefetch(4))
        return dataset
    else:
        return None
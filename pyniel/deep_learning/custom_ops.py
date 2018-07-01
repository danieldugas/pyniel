import numpy as np
import tensorflow as tf


def n_dimensional_weightmul(L, W, L_shape, Lout_shape, first_dim_of_l_is_batch=True):
    """ Equivalent to matmul(W,L)
      but works for L with larger shapes than 1
      L_shape and Lout_shape are excluding the batch dimension (0)"""
    if not first_dim_of_l_is_batch:
        raise NotImplementedError
    if len(L_shape) == 1 and len(Lout_shape) == 1:
        return tf.matmul(L, W)
    # L    : ?xN1xN2xN3x...
    # Lout : ?xM1xM2xM3x...
    # W    : N1xN2x...xM1xM2x...
    # Einstein notation: letter b (denotes batch dimension)
    # Lout_blmn... = L_bijk... * Wijk...lmn...
    letters = list("ijklmnopqrst")
    l_subscripts = "".join([letters.pop(0) for _ in range(len(L_shape))])
    lout_subscripts = "".join([letters.pop(0) for _ in range(len(Lout_shape))])
    einsum_string = (
        "b"
        + l_subscripts
        + ","
        + l_subscripts
        + lout_subscripts
        + "->"
        + "b"
        + lout_subscripts
    )
    return tf.einsum(einsum_string, L, W)


def gaussian_log_likelihood(sample, mean, log_sigma_squared):
    """
    gaussian log likelihood in tensorflow
    """
    TINY = 1e-8  # Do this better
    stddev = tf.sqrt(tf.exp(log_sigma_squared))
    epsilon = (sample - mean) / (stddev + TINY)
    return tf.reduce_sum(
        -0.5 * np.log(2 * np.pi) - tf.log(stddev + TINY) - 0.5 * tf.square(epsilon),
        axis=1,
    )

import tensorflow as tf
import numpy as np


def gs(x):
    return x.get_shape().as_list()


def reparam_gamma(a, b):
    """

    :param a: shape
    :param b: inverse scale
    :return: differentiable reparameterization of gamma(a, b)
    """
    assert gs(a) == gs(b), "shape must be same"
    z = tf.random_gamma(tf.shape(a), a, b)
    digamma_a = tf.digamma(a)
    logb = tf.log(b)
    sqrt_digamma_1_a = tf.sqrt(tf.polygamma(1.0, a))
    num = tf.log(z) - digamma_a + logb
    den = sqrt_digamma_1_a
    eps = tf.stop_gradient(num / den)
    logz = (eps * sqrt_digamma_1_a) - logb + digamma_a
    z_reparam = tf.exp(logz)
    return z_reparam

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    sess = tf.Session()
    alpha = 1.
    beta = 1.


    a = tf.constant([alpha], dtype=tf.float32)
    b = tf.constant([beta], dtype=tf.float32)
    z = reparam_gamma(a, b)

    numpy_gamma = [np.random.gamma(alpha, beta) for i in range(2000)]
    plt.hist(numpy_gamma, 100, alpha=.5, label='numpy')

    get = lambda: sess.run(z)[0][0]
    tf_gamma = [get() for i in range(2000)]
    plt.hist(tf_gamma, 100, alpha=.5, label='tf')

    plt.show()




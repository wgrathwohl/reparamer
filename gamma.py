import tensorflow as tf
import numpy as np


def gs(x):
    return x.get_shape().as_list()


def sanity_test():
    size = (50000,)
    shape_parameter = .1
    scale_parameter = 0.5
    bins = np.linspace(-1, 5, 30)

    np_res = np.random.gamma(shape=shape_parameter, scale=scale_parameter, size=size)

    # Note the 1/scale_parameter here

    tf_op = tf.random_gamma(shape=size, alpha=shape_parameter, beta=1 / scale_parameter)
    with tf.Session() as sess:
        tf_res = sess.run(tf_op)

    plt.hist(tf_res, bins=bins, alpha=0.5)
    plt.hist(np_res, bins=bins, alpha=0.5)
    plt.show()


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
    return eps
    logz = (eps * sqrt_digamma_1_a) - logb + digamma_a
    z_reparam = tf.exp(logz)
    return z_reparam


# Log density of Ga(alpha, beta)
def log_q(z, alpha, beta):
    return -tf.lgamma(alpha) + alpha * tf.log(beta) \
           + (alpha - 1.) * tf.log(z) - beta * z


# Log density of N(0, 1)
def log_s(epsilon):
    return -0.5 * np.log(2. * np.pi) - 0.5 * tf.square(epsilon)


# Transformation and its derivative
# Transforms eps ~ N(0, 1) to proposal distribution
def h(epsilon, alpha, beta):
    assert len(gs(epsilon)) == 2
    assert gs(alpha) == gs(beta)
    return (alpha - 1. / 3.) * (1. + epsilon / tf.sqrt(9. * alpha - 3.))**3. / beta


def dh(epsilon, alpha, beta):
    return (alpha - 1. / 3.) * 3. / tf.sqrt(9. * alpha - 3.) * \
           (1. + epsilon / tf.sqrt(9. * alpha - 3.))**2. / beta


def h_inverse(z, alpha, beta):
    return tf.sqrt(9.0 * alpha - 3) * ((beta * z / (alpha - 1./3))**(1./3) - 1)


# Log density of proposal r(z) = s(epsilon) * |dh/depsilon|^{-1}
def log_r(epsilon, alpha, beta):
    return -tf.log(dh(epsilon, alpha, beta)) + log_s(epsilon)


# Density of the accepted value of epsilon
# (this is just a change of variables too)
def log_pi(epsilon, alpha, beta):
    return log_s(epsilon) + \
           log_q(h(epsilon, alpha, beta), alpha, beta) - \
           log_r(epsilon, alpha, beta)


def gamma_entropy(alpha, beta):
    return alpha - tf.log(beta) + tf.lgamma(alpha) + \
           (1. - alpha) * tf.digamma(alpha)

def sample_pi(alpha, beta, size=(1,)):
    gamma_samples = tf.random_gamma(size, alpha, beta)
    return h_inverse(gamma_samples, alpha, beta)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # follow experiment presented in paper and code
    z_true = 3.0
    a0, b0 = 1.0, 1.0
    N = 1000
    x = np.random.poisson(z_true, size=N).astype(np.float32)[:, np.newaxis]
    plt.hist(x)
    plt.show()
    x = tf.constant(x, dtype=tf.float32)

    def log_p(x, z):
        pz = -tf.lgamma(a0) + a0 * tf.log(b0) + (a0 - 1.) * z - b0 * z
        pxgz = -tf.lgamma(x + 1.) - z + x * tf.log(z)
        assert gs(pz) == gs(pxgz)
        return pz + pxgz


    # We can compute the true posterior in closed form
    alpha_true = a0 + tf.reduce_sum(x)
    beta_true = tf.constant(b0 + N, dtype=tf.float32)

    _alpha_param = tf.get_variable(
        "alpha_param", [1],
        dtype=tf.float32, initializer=tf.constant_initializer(2.)
    )
    _beta_param = tf.get_variable(
        "beta_param", [1],
        dtype=tf.float32, initializer=tf.constant_initializer(2.)
    )

    alpha = tf.exp(_alpha_param) + 1.
    beta = tf.exp(_beta_param)

    # sample epsilon from gamma -> h^-1 -> epsilon
    epsilon = sample_pi(alpha, beta, (N,))
    z = h(epsilon, alpha, beta)
    # get per-sample objective
    f_i = log_p(x, z) + gamma_entropy(alpha, beta)
    # get per-sample log-likelihoods
    lp = log_pi(epsilon, alpha, beta)
    # total obj = reparm + no_grad(reparam) * logpi
    elbo = tf.reduce_mean(f_i)
    f = elbo + tf.reduce_mean(tf.stop_gradient(f_i) * lp)
    loss = -f

    opt = tf.train.AdamOptimizer(.01)
    opt_op = opt.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        at, bt = sess.run([alpha_true, beta_true])
        _fi, = sess.run([elbo])
        for i in range(10000):
            l, alpha_star, beta_star, _elbo, _ = sess.run([loss, alpha, beta, elbo, opt_op])
            if i % 100 == 0:
                print("true a = ", at)
                print("infd a = ", alpha_star)
                print("true b = ", bt)
                print("infd b = ", beta_star)
                print("elbo (loss) = {} ({})".format(_elbo, l))

        import scipy.stats
        zs = np.linspace(0, 6, 100)
        plt.plot(zs, scipy.stats.gamma(at, scale=1. / bt).pdf(zs), label="true post.")
        plt.plot(zs, scipy.stats.gamma(alpha_star[0], scale=1. / beta_star[0]).pdf(zs), label="var. post.")
        plt.legend(loc="upper right")
        plt.xlabel("$z$")
        plt.ylabel("$p(z \\mid x)$")
        plt.show()







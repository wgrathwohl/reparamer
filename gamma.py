import tensorflow as tf
import numpy as np


def gs(x):
    return x.get_shape().as_list()


def log(x, eps=1e-8):
    return tf.log(x + eps)

# Log density of Ga(alpha, beta)
def gamma_logpdf(z, alpha, beta):
    return -tf.lgamma(alpha) + alpha * log(beta) \
           + (alpha - 1.) * log(z) - beta * z


log_q = gamma_logpdf


# Log density of N(0, 1)
def log_s(epsilon):
    return -0.5 * np.log(2. * np.pi) - 0.5 * tf.square(epsilon)


# Transformation and its derivative
# Transforms eps ~ N(0, 1) to proposal distribution
def h(epsilon, alpha, beta):
    assert len(gs(epsilon)) == 2
    assert gs(alpha) == gs(beta)
    z = (alpha - 1. / 3.) * (1. + epsilon / tf.sqrt(9. * alpha - 3.))**3. / beta
    return z


def shape_augmentation(z_tilde, B, alpha):
    logz = log(z_tilde)
    for i in range(1, B + 1):
        u = tf.random_uniform(tf.shape(z_tilde))
        logz = logz + log(u) / (alpha + i - 1.)
    return tf.exp(logz)


def dh(epsilon, alpha, beta):
    return (alpha - 1. / 3.) * 3. / tf.sqrt(9. * alpha - 3.) * \
           (1. + epsilon / tf.sqrt(9. * alpha - 3.))**2. / beta


def h_inverse(z, alpha, beta):
    return tf.sqrt(9.0 * alpha - 3) * ((beta * z / (alpha - 1./3))**(1./3) - 1)


# Log density of proposal r(z) = s(epsilon) * |dh/depsilon|^{-1}
def log_r(epsilon, alpha, beta):
    return -log(dh(epsilon, alpha, beta)) + log_s(epsilon)


# Density of the accepted value of epsilon
# (this is just a change of variables too)
# This ignores the s(epsilon) term since it drops in the gradient
def log_pi(epsilon, alpha, beta, z=None):
    if z is None:
        lq = log_q(h(epsilon, alpha, beta), alpha, beta)
    else:
        lq = log_q(z, alpha, beta)
    return lq - log_r(epsilon, alpha, beta)


def gamma_entropy(alpha, beta):
    return alpha - log(beta) + tf.lgamma(alpha) + (1. - alpha) * tf.digamma(alpha)


def sample_pi(alpha, beta, size=(1,)):
    gamma_samples = tf.random_gamma(size, alpha, beta)
    return tf.stop_gradient(h_inverse(gamma_samples, alpha, beta))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # follow experiment presented in paper and code
    z_true = 3.0
    a0, b0 = 1.0, 1.0
    N = 10
    batch_size = 10
    #x = np.random.poisson(z_true, size=N).astype(np.float32)[np.newaxis, :]
    x = np.array([5., 3., 6., 2., 5., 4., 2., 1., 5., 2.])
    x = x[np.newaxis, :]
    x = tf.constant(x, dtype=tf.float32)

    def log_p(x, z):
        pz = gamma_logpdf(z, a0, b0)
        pxgz = -tf.lgamma(x + 1.) - z + x * log(z)
        return pz + tf.reduce_sum(pxgz, axis=1, keep_dims=True)


    B = 5  # for shape augmentation

    # We can compute the true posterior in closed form
    alpha_true = a0 + tf.reduce_sum(x)
    beta_true = tf.constant(b0 + N, dtype=tf.float32)

    _alpha_param = tf.get_variable(
        "alpha_param", [1],
        dtype=tf.float32, initializer=tf.constant_initializer(np.log(2.)), trainable=True
    )
    _beta_param = tf.get_variable(
        "beta_param", [1],
        dtype=tf.float32, initializer=tf.constant_initializer(np.log(2.)), trainable=True
    )

    alpha = tf.exp(_alpha_param)
    beta = tf.exp(_beta_param)

    # sample epsilon from gamma -> h^-1 -> epsilon
    epsilon = sample_pi(alpha + B, beta, (batch_size,))
    z_tilde = h(epsilon, alpha + B, beta)
    z = shape_augmentation(z_tilde, B, alpha)
    # get per-sample objective
    ent = gamma_entropy(alpha, beta)[0]
    likelihood_i = log_p(x, z)
    likelihood = tf.reduce_mean(likelihood_i)
    # get per-sample log-likelihoods
    lp = log_pi(epsilon, alpha + B, beta, z=z_tilde)
    elbo = likelihood + ent
    score_obj = tf.reduce_mean(tf.stop_gradient(likelihood_i) * lp)
    reparam_obj = elbo
    # total obj = reparm + no_grad(reparam) * logpi
    f = reparam_obj + score_obj
    loss = -f
    opt = tf.train.MomentumOptimizer(.01, .9, use_nesterov=True)
    opt_op = opt.minimize(loss, var_list=[_alpha_param, _beta_param])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        at, bt = sess.run([alpha_true, beta_true])

        elbos = []
        for i in range(300):
            _elbo, _ = sess.run([elbo, opt_op])
            elbos.append(_elbo)

        alpha_star, beta_star = sess.run([alpha, beta])
        print("true a = ", at)
        print("infd a = ", alpha_star[0])
        print("true b = ", bt)
        print("infd b = ", beta_star[0])
        print("elbo = {}".format(_elbo))
        print("E_q(z; theta)[z] = ", alpha_star[0] / beta_star[0])

        plt.plot(elbos)
        plt.xlabel("Iteration")
        plt.ylabel("ELBO")
        plt.show()

        import scipy.stats
        zs = np.linspace(0, 6, 100)
        plt.plot(zs, scipy.stats.gamma(at, scale=1. / bt).pdf(zs), label="true post.")
        plt.plot(zs, scipy.stats.gamma(alpha_star[0], scale=1. / beta_star[0]).pdf(zs), label="var. post.")
        plt.legend(loc="upper right")
        plt.xlabel("$z$")
        plt.ylabel("$p(z \\mid x)$")
        plt.show()







import keras.backend as K
from attack_utils import gen_grad, gen_hessian
import tensorflow as tf

def symbolic_second_ord(x, grad, hessian, dim, eps=0.3, clipping=True):

    hessian_inv = tf.matrix_inverse(hessian)

    grad = tf.reshape(grad, [dim, 1])

    perturb = 2 * tf.matmul(hessian_inv, grad)

    perturb = tf.reshape(perturb, [dim])

    normed_perturb = K.sign(perturb)

    scaled_perturb = eps * normed_perturb

    adv_x = K.stop_gradient(x + scaled_perturb)

    if clipping:
        adv_x = K.clip(adv_x, 0, 1)

    return adv_x


def symbolic_fgs(x, grad, eps=0.3, clipping=True):
    """
    FGSM attack.
    """

    # signed gradient
    normed_grad = K.sign(grad)

    # Multiply by constant epsilon
    scaled_grad = eps * normed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = K.stop_gradient(x + scaled_grad)

    if clipping:
        adv_x = K.clip(adv_x, 0, 1)
    return adv_x

def symbolic_fg(x, grad, eps=0.3, clipping=True):
    """
    FG attack
    """
    # Unit vector in direction of gradient
    reduc_ind = list(xrange(1, len(x.get_shape())))
    normed_grad = grad / tf.sqrt(tf.reduce_sum(tf.square(grad),
                                                   reduction_indices=reduc_ind,
                                                   keep_dims=True))
    # Multiply by constant epsilon
    scaled_grad = eps * normed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = K.stop_gradient(x + scaled_grad)

    if clipping:
        adv_x = K.clip(adv_x, 0, 1)

    return adv_x


def iter_fgs(model, x, y, steps, alpha, eps, clipping=True):
    """
    I-FGSM attack.
    """

    adv_x = x
    # iteratively apply the FGSM with small step size
    for i in range(steps):
        logits = model(adv_x)
        grad = gen_grad(adv_x, logits, y)

        adv_x = symbolic_fgs(adv_x, grad, alpha, True)
        r = adv_x - x
        r = K.clip(r, -eps, eps)
        adv_x = x+r

    if clipping:
        adv_x = K.clip(adv_x, 0, 1)


    return adv_x

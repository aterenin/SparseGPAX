
from multipledispatch import dispatch
import jax.random as jr
import tensorflow_probability
tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels


@dispatch(tfk.ExponentiatedQuadratic)
def spectral_measure(k, input_dimension, num_samples, key):
    """Draws samples from the kernel's spectral measure.

    Args:
        input_dimension: the kernel's input space dimension.
        num_samples: the number of samples to draw.
        key: the random number generator key.
    """
    return jr.normal(key, (num_samples,1,input_dimension))


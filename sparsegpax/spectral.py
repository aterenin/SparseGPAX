from typing import Tuple
from multipledispatch import dispatch
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability
tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels


@dispatch(tfk.ExponentiatedQuadratic, int, int, object)
def standard_spectral_measure(kernel, input_dimension, num_samples, key) -> jnp.ndarray:
    """Draws samples from the kernel's spectral measure.

    Args:
        kernel: the kernel.
        input_dimension: the kernel's input space dimension.
        num_samples: the number of samples to draw.
        key: the random number generator key.
    """
    return jr.normal(key, (num_samples,1,input_dimension))


@dispatch(tfk.ExponentiatedQuadratic, int, object)
def spectral_weights(kernel, input_dimension, frequency) -> Tuple[jnp.ndarray,jnp.ndarray]:
    """Computes the input weights and output weights associated with the kernel.

    Args:
        kernel: the kernel.
        input_dimension: the kernel's input space dimension.
        frequency: the sampled frequencies.
    """
    return (kernel.amplitude, jnp.ones((input_dimension)))


@dispatch(tfk.FeatureScaled, int, int, object)
def standard_spectral_measure(kernel, input_dimension, num_samples, key) -> jnp.ndarray:
    """Draws samples from the kernel's spectral measure.

    Args:
        kernel: the kernel.
        input_dimension: the kernel's input space dimension.
        num_samples: the number of samples to draw.
        key: the random number generator key.
    """
    return standard_spectral_measure(kernel.kernel, input_dimension, num_samples, key)


@dispatch(tfk.FeatureScaled, int, object)
def spectral_weights(kernel, input_dimension, frequency) -> Tuple[jnp.ndarray,jnp.ndarray]:
    """Computes the input weights and output weights associated with the kernel.

    Args:
        kernel: the kernel.
        input_dimension: the kernel's input space dimension.
        frequency: the sampled frequencies.
    """
    (parent_outer_weights, parent_inner_weights) = spectral_weights(kernel.kernel, input_dimension, frequency)
    return (parent_outer_weights, parent_inner_weights * kernel.scale_diag)

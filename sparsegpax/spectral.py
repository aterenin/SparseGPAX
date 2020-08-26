from typing import Tuple
# from multipledispatch import dispatch
import jax.numpy as jnp
import jax.random as jr
from jax import jit
import tensorflow_probability
tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels


# @dispatch(tfk.ExponentiatedQuadratic, int, int, int, object)
def standard_spectral_measure(kernel, output_dimension, input_dimension, num_samples, key) -> jnp.ndarray:
    """Draws samples from the kernel's spectral measure.

    Args:
        kernel: the kernel.
        output_dimension: the kernel's output dimension.
        input_dimension: the kernel's input space dimension.
        num_samples: the number of samples to draw.
        key: the random number generator key.
    """
    return jr.normal(key, (output_dimension, input_dimension, num_samples))


# @dispatch(tfk.ExponentiatedQuadratic, object)
def spectral_weights(kernel, frequency) -> Tuple[jnp.ndarray,jnp.ndarray]:
    """Computes the input weights and output weights associated with the kernel.

    Args:
        kernel: the kernel.
        frequency: the sampled frequencies.
    """
    (output_dimension, input_dimension, num_samples) = frequency.shape
    amplitude = kernel.amplitude if kernel.amplitude is not None else jnp.ones((output_dimension,num_samples))
    return (amplitude, jnp.ones((input_dimension,)))



# @dispatch(tfk.FeatureScaled, int, int, int, object)
# def standard_spectral_measure(kernel, output_dimension, input_dimension, num_samples, key) -> jnp.ndarray:
#     """Draws samples from the kernel's spectral measure.

#     Args:
#         kernel: the kernel.
#         output_dimension: the kernel's output dimension.
#         input_dimension: the kernel's input space dimension.
#         num_samples: the number of samples to draw.
#         key: the random number generator key.
#     """
#     return standard_spectral_measure(kernel.kernel, output_dimension, input_dimension, num_samples, key)


# @dispatch(tfk.FeatureScaled, object)
# def spectral_weights(kernel, frequency) -> Tuple[jnp.ndarray,jnp.ndarray]:
#     """Computes the input weights and output weights associated with the kernel.

#     Args:
#         kernel: the kernel.
#         frequency: the sampled frequencies.
#     """
#     (parent_outer_weights, parent_inner_weights) = spectral_weights(kernel.kernel, frequency)
#     return (parent_outer_weights, parent_inner_weights * kernel.scale_diag)

from abc import ABCMeta
from typing import Tuple, Optional
import haiku as hk
import jax.numpy as jnp
import jax.random as jr
from jax import jit
import tensorflow_probability
tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels

class ScaledKernel(hk.Module):
    """A kernel with learned amplitude and length scale parameters.

    """
    def __init__(
        self,
        kernel_class: ABCMeta,
        input_dimension: int,
        output_dimension: int,
        name: Optional[str] = None,
    ):
        """Scales the given kernel input by length scales and output by amplitudes.

        Args:
            kernel_class: the class of the covariance kernel.
            input_dimension: the input space dimension.
            output_dimension: the output space dimension.
            name: the Haiku module name.
        """
        super().__init__(name=name)
        self.kernel_class = kernel_class
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        hk.get_parameter("log_amplitudes", [self.output_dimension], init=jnp.zeros)
        hk.get_parameter("log_length_scales", [self.input_dimension], init=jnp.zeros)

    def matrix(
        self,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        """Assemble the kernel matrix.

        Args:
            x1: the first input.
            x2: the second input.
        """
        return self.kernel().matrix(x1,x2)

    def kernel(
        self,
    ):
        """Instantiates the kernel with the given parameters.
        """
        amplitudes = jnp.exp(hk.get_parameter("log_amplitudes", [self.output_dimension], init=jnp.zeros))
        length_scales = jnp.exp(hk.get_parameter("log_length_scales", [self.input_dimension], init=jnp.zeros))
        return self.kernel_class(amplitude = amplitudes, length_scale = length_scales)

    def standard_spectral_measure(
        self,
        num_samples: int
    ) -> jnp.ndarray:
        """Draws samples from the kernel's spectral measure.

        Args:
            num_samples: the number of samples to draw.
        """
        if self.kernel_class == tfk.ExponentiatedQuadratic:
            return jr.normal(hk.next_rng_key(), (self.output_dimension, self.input_dimension, num_samples))
        else: 
            raise Exception("Spectral measure not implemented for this kernel.")

    def spectral_weights(
        self,
        frequency: jnp.ndarray,
    ) -> Tuple[jnp.ndarray,jnp.ndarray]: 
        """Computes the input weights and output weights associated with the kernel.

        Args:
            kernel: the kernel.
            frequency: the sampled frequencies.
        """
        amplitudes = jnp.exp(hk.get_parameter("log_amplitudes", [self.output_dimension], init=jnp.zeros))
        length_scales = jnp.exp(hk.get_parameter("log_length_scales", [self.input_dimension], init=jnp.zeros))
        return (amplitudes, length_scales)
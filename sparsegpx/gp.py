from typing import Optional
import haiku as hk
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import tensorflow_probability
tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels
from sparsegpx.spectral import spectral_measure

class SparseGaussianProcess(): 
    """A sparse Gaussian process, implemented as a Haiku module

    """
    def __init__(
            self,
            input_dimension: int,
            kernel: tfk.PositiveSemidefiniteKernel,
            key: jr.PRNGKey,
            name: Optional[str] = None,
    ):
        """Initializes the sparse GP.

        Args:
            name: module name.
        """
        # super().__init__(name=name)
        num_basis = 64
        num_samples = 8
        num_inducing = 16
        self.input_dimension = input_dimension
        self.output_dimension  = 1
        self.kernel = kernel
        self.prior_frequency = jnp.zeros((num_basis, self.output_dimension, self.input_dimension))
        self.prior_phase = jnp.zeros((num_basis, self.output_dimension))
        self.prior_weights = jnp.zeros((num_samples, num_basis))
        self.inducing_locations = jr.normal(key, (num_inducing, self.input_dimension))
        self.inducing_pseudo_mean = jnp.zeros((num_inducing*self.output_dimension))
        self.inducing_pseudo_log_errvar = jnp.ones((num_inducing*self.output_dimension))
        self.inducing_weights = jnp.zeros((num_inducing,))
        self.cholesky_cache = jsp.linalg.cholesky(kernel.matrix(self.inducing_locations, self.inducing_locations) + jnp.diag(jnp.exp(self.inducing_pseudo_log_errvar)), lower=True)
        self.log_error = jnp.zeros((1,))
        self.resample_prior_basis(num_basis,key)
        self.randomize(num_samples,key)


    def __call__(
            self,
            x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluates the sparse GP for a given input matrix.

        Args:
            x: the input matrix.
        """
        return x


    def randomize(
            self,
            num_samples: int,
            key: jr.PRNGKey,
    ):
        """Samples a new set of random functions from the GP.

        Args:
            num_basis: the number of samples to draw.
            key: the random number generator key.
        """
        self.prior_weights = jr.normal(key, (num_samples,self.prior_weights.shape[-1]))


    def resample_prior_basis(
            self,
            num_basis: int,
            key: jr.PRNGKey,
    ):
        """Resamples the frequency and phase of the prior random feature basis.

        Args:
            num_basis: the number of basis functions to use.
            key: the random number generator key.
        """
        self.prior_frequency = spectral_measure(self.kernel, self.input_dimension, num_basis, key)
        self.prior_phase = jr.uniform(key, (num_basis, self.output_dimension), maxval=2*jnp.pi)
        self.randomize(self.prior_weights.shape[-1],key)


    def prior(
            self,
            x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluates the prior GP at x.

        Args:
            x: the input matrix.
        """
        (L,OD,ID) = self.prior_frequency.shape
        (N,ID2) = x.shape
        (S,L2) = self.prior_weights.shape
        assert L==L2
        assert ID==ID2
        basis_fn_inner_prod = jnp.reshape(jnp.matmul(jnp.reshape(self.prior_frequency, (L*OD,ID)), x.T), (L,OD,N))
        basis_fn = jnp.cos(basis_fn_inner_prod + jnp.reshape(self.prior_phase, (L,OD,1)))
        basis_weight = jnp.sqrt(2/L) * self.prior_weights # TODO: typecast this to f32!
        output = jnp.reshape(jnp.matmul(basis_weight, jnp.reshape(jnp.transpose(basis_fn, (1,0,2)), (L,N*OD))), (S,N,OD))
        return output


    def prior_KL(
            self
    ) -> jnp.ndarray:
        """Evaluates the prior KL term in the sparse VI objective.

        """
        return jnp.zeros(1)

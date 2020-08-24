from typing import Optional
import haiku as hk
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax import vmap
import tensorflow_probability
tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels
from sparsegpax.spectral import standard_spectral_measure, spectral_weights

class SparseGaussianProcess: 
    """A sparse Gaussian process, implemented as a Haiku module.
    """
    def __init__(
            self,
            input_dimension: int,
            output_dimension: int,
            kernel: tfk.PositiveSemidefiniteKernel,
            key: jr.PRNGKey,
            num_basis: int = 64,
            num_samples: int = 8,
            num_inducing: int = 16,
    ):
        """Initializes the sparse GP.

        Args:
            name: module name.
        """
        key_inducing_loc, key_prior = jr.split(key)
        self.kernel = kernel
        self.prior_frequency = jnp.zeros((output_dimension, input_dimension, num_basis))
        self.prior_phase = jnp.zeros((output_dimension, num_basis))
        self.prior_weights = jnp.zeros((output_dimension, num_basis, num_samples))
        self.inducing_locations = jr.normal(key_inducing_loc, (num_inducing, input_dimension))
        self.inducing_pseudo_mean = jnp.zeros((output_dimension, num_inducing))
        self.inducing_pseudo_log_errvar = jnp.ones((output_dimension, num_inducing))
        self.inducing_weights = jnp.zeros((output_dimension,num_inducing,num_samples))
        self.cholesky = jnp.zeros((output_dimension,num_inducing,num_inducing))
        self.log_error = jnp.zeros((output_dimension,))
        self.resample_prior_basis(num_basis, key_prior)


    def __call__(
            self,
            x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluates the sparse GP for a given input matrix.

        Args:
            x: the input matrix.
        """
        (N,ID) = x.shape
        (OD,M,S) = self.inducing_weights.shape

        # evaluate the prior part at x
        f_prior = self.prior(x)
        assert f_prior.shape == (S,N,OD)

        # evaluate the data part at x
        f_data = (self.kernel.matrix(x, self.inducing_locations) @ self.inducing_weights).T # non-batched
        assert f_data.shape == (S,N,OD)

        # combine
        return f_prior + f_data


    def randomize(
            self,
            num_samples: int,
            key: jr.PRNGKey,
    ):
        """Samples a new set of random functions from the GP.

        Args:
            num_samples: the number of samples to draw.
            key: the random number generator key.
        """
        # sample the prior weights w_i ~ N(0,1) IID
        (OD,L,S) = self.prior_weights.shape
        key_prior_weights, key_residual = jr.split(key)
        self.prior_weights = jr.normal(key_prior_weights, (OD,L,num_samples))
        assert self.prior_weights.shape == (OD,L,num_samples)
        
        # compute the mean-reparameterized inducing weights v = \mu + (K + V)^{-1}(f - \eps)
        (M,ID) = self.inducing_locations.shape
        (OD2,M2,S2) = self.inducing_weights.shape
        assert M==M2
        assert OD==OD2
        assert S==S2
        (self.cholesky,lower) = jsp.linalg.cho_factor(self.kernel.matrix(self.inducing_locations, self.inducing_locations) + vmap(jnp.diag)(jnp.exp(self.inducing_pseudo_log_errvar)))
        assert self.cholesky.shape==(OD,M,M)
        assert lower==False
        prior = self.prior(self.inducing_locations)
        assert prior.shape==(S,M,OD)
        residual = prior.T - (jnp.reshape(jnp.exp(self.inducing_pseudo_log_errvar / 2), (OD,M,1)) * jr.normal(key_residual, (OD,M,num_samples))) # TODO: careful with f32!
        assert residual.shape==(OD,M,num_samples)
        self.inducing_weights = jnp.reshape(self.inducing_pseudo_mean,(OD,M,1)) + jsp.linalg.cho_solve((self.cholesky,False),residual)
        assert self.inducing_weights.shape==(OD,M,num_samples)


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
        key_frequency, key_phase, key_randomize = jr.split(key, 3)
        (OD,ID,L) = self.prior_frequency.shape
        self.prior_frequency = standard_spectral_measure(self.kernel, OD, ID, num_basis, key_frequency)
        assert self.prior_frequency.shape == (OD,ID,num_basis)
        (OD2,L2) = self.prior_phase.shape
        assert OD2==OD
        assert L2==L
        self.prior_phase = jr.uniform(key_phase, (OD, num_basis), maxval=2*jnp.pi)
        assert self.prior_phase.shape == (OD,num_basis)
        (OD3,L3,S) = self.prior_weights.shape
        assert OD3==OD
        assert L3==L2
        self.randomize(S,key_randomize)


    def prior(
            self,
            x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluates the prior GP at x.

        Args:
            x: the input matrix.
        """
        (OD,ID,L) = self.prior_frequency.shape
        (OD2,L2) = self.prior_phase.shape
        (N,ID2) = x.shape
        (OD3,L3,S) = self.prior_weights.shape
        (outer_weights, inner_weights) = spectral_weights(self.kernel, self.prior_frequency)
        (OD4,L4) = outer_weights.shape
        (ID3,) = inner_weights.shape
        assert L==L2
        assert L==L3
        assert L==L4
        assert ID==ID2
        assert OD==OD2
        assert ID==ID3
        assert OD==OD3
        assert OD==OD4
        rescaled_x = x / inner_weights
        assert rescaled_x.shape == (N,ID)
        basis_fn_inner_prod = rescaled_x @ self.prior_frequency
        assert basis_fn_inner_prod.shape == (OD,N,L)
        basis_fn = jnp.cos(basis_fn_inner_prod + jnp.reshape(self.prior_phase, (OD,1,L)))
        assert basis_fn.shape == (OD,N,L)
        basis_weight = jnp.sqrt(2/L) * jnp.reshape(outer_weights,(OD,L,1)) * self.prior_weights # TODO: typecast this to f32!
        assert basis_weight.shape == (OD,L,S)
        output = (basis_fn @ basis_weight).T
        assert output.shape == (S,N,OD)
        return output


    def prior_KL(
            self,
    ) -> jnp.ndarray:
        """Evaluates the prior KL term in the sparse VI objective. 
        
        """
        logdet_term = (2 * jnp.sum(vmap(jnp.diag)(self.cholesky))) - jnp.sum(self.inducing_pseudo_log_errvar)
        kernel_matrix = self.kernel.matrix(self.inducing_locations, self.inducing_locations)
        cholesky_inv = vmap(lambda x: jsp.linalg.cho_solve((x,False), jnp.eye(*x.shape)))(self.cholesky)
        trace_term = jnp.sum(cholesky_inv * kernel_matrix)
        reparameterized_quadratic_form_term = jnp.sum(self.inducing_pseudo_mean @ kernel_matrix @ self.inducing_pseudo_mean.T)
        return (logdet_term - self.inducing_pseudo_mean.shape[0] + trace_term + reparameterized_quadratic_form_term) / 2

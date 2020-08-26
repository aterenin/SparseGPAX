from typing import Optional
import haiku as hk
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import jax
import tensorflow_probability
from tensorflow_probability.python.internal.backend import jax as tf2jax
tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels
from sparsegpax.spectral import standard_spectral_measure, spectral_weights

class SparseGaussianProcess(hk.Module): 
    """A sparse Gaussian process, implemented as a Haiku module.

    """


    def __init__(
            self,
            kernel: tfk.PositiveSemidefiniteKernel,
            input_dimension: int,
            output_dimension: int,
            num_inducing: int,
            num_basis: int,
            num_samples: int,
            name: Optional[str] = None,
    ):
        """Initializes the sparse GP.

        Args:
            kernel: the covariance kernel.
            input_dimension: the input space dimension.
            output_dimension: the output space dimension.
            num_inducing: the number of inducing points per input dimension.
            num_basis: the number of prior basis functions.
            num_samples: the number of samples stored in the GP.
            name: the module name.
        """
        super().__init__(name=name)
        self.kernel = kernel
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.num_inducing = num_inducing
        self.num_basis = num_basis
        self.num_samples = num_samples

        self.resample_prior_basis()
        self.randomize()


    def __call__(
            self,
            x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluates the sparse GP for a given input matrix.

        Args:
            x: the input matrix.
        """
        (S,OD,ID,M) = (self.num_samples, self.output_dimension, self.input_dimension, self.num_inducing)
        inducing_locations = hk.get_parameter("inducing_locations", [M,ID], init=hk.initializers.RandomUniform())
        inducing_weights = hk.get_state("inducing_weights", [S,OD,M], init=jnp.zeros)

        f_prior = self.prior(x)
        f_data = tf2jax.linalg.matvec(self.kernel.matrix(x, inducing_locations), inducing_weights) # non-batched

        (N,ID) = x.shape
        (S,OD,M) = inducing_weights.shape
        assert f_prior.shape == (S,OD,N)
        assert f_data.shape == (S,OD,N)

        return f_prior + f_data


    def randomize(
            self,
    ):
        """Samples a new set of random functions from the GP.

        """
        (S,OD,ID,M,L) = (self.num_samples, self.output_dimension, self.input_dimension, self.num_inducing, self.num_basis)
        inducing_locations = hk.get_parameter("inducing_locations", [M,ID], init=hk.initializers.RandomUniform())
        inducing_pseudo_mean = hk.get_parameter("inducing_pseudo_mean", [OD,M], init=jnp.zeros)
        inducing_pseudo_log_errvar = hk.get_parameter("inducing_pseudo_log_errvar", [OD,M], init=jnp.zeros)
        
        prior_weights = jr.normal(hk.next_rng_key(), (S,OD,L))
        hk.set_state("prior_weights", prior_weights)

        (cholesky,_) = jsp.linalg.cho_factor(self.kernel.matrix(inducing_locations, inducing_locations) + jax.vmap(jnp.diag)(jnp.exp(inducing_pseudo_log_errvar)), lower=True)
        residual = self.prior(inducing_locations) - jnp.exp(inducing_pseudo_log_errvar / 2) * jr.normal(hk.next_rng_key(),(S,OD,M)) # TODO: careful with f32!
        inducing_weights = inducing_pseudo_mean + tf2jax.linalg.LinearOperatorLowerTriangular(cholesky).solvevec(residual) # mean-reparameterized v = \mu + (K + V)^{-1}(f - \eps)

        hk.set_state("inducing_weights", inducing_weights)
        hk.set_state("cholesky", cholesky)
        
        assert prior_weights.shape == (S,OD,L)
        (M,ID) = inducing_locations.shape
        (S,OD2,M2) = inducing_weights.shape
        assert M==M2
        assert OD==OD2
        assert cholesky.shape==(OD,M,M)
        assert self.prior(inducing_locations).shape==(S,OD,M)
        assert residual.shape==(S,OD,M)
        assert inducing_weights.shape==(S,OD,M)


    def resample_prior_basis(
            self,
    ):
        """Resamples the frequency and phase of the prior random feature basis.

        """
        (OD,ID,L) = (self.output_dimension, self.input_dimension, self.num_basis)
        prior_frequency = standard_spectral_measure(self.kernel, OD, ID, L, hk.next_rng_key())
        prior_phase = jr.uniform(hk.next_rng_key(), (OD, L), maxval=2*jnp.pi)

        hk.set_state("prior_frequency", prior_frequency)
        hk.set_state("prior_phase", prior_phase)

        (OD,ID,L) = prior_frequency.shape
        assert prior_frequency.shape == (OD,ID,L)
        (OD2,L2) = prior_phase.shape
        assert OD2==OD
        assert L2==L
        assert prior_phase.shape == (OD,L)


    def prior(
            self,
            x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluates the prior GP at x.

        Args:
            x: the input matrix.
        """
        (S,OD,ID,L) = (self.num_samples, self.output_dimension, self.input_dimension, self.num_basis)
        prior_frequency = hk.get_state("prior_frequency", [OD,ID,L], init=jnp.zeros)
        prior_phase = hk.get_state("prior_phase", [OD,L], init=jnp.zeros)
        prior_weights = hk.get_state("prior_weights", [S,OD,L], init=jnp.zeros)
        
        (outer_weights, inner_weights) = spectral_weights(self.kernel, prior_frequency)
        rescaled_x = x / inner_weights
        basis_fn_inner_prod = rescaled_x @ prior_frequency
        basis_fn = jnp.cos(basis_fn_inner_prod + jnp.expand_dims(prior_phase,-2))
        basis_weight = jnp.sqrt(2/L) * outer_weights * prior_weights
        output = tf2jax.linalg.matvec(basis_fn, basis_weight)

        (OD,ID,L) = prior_frequency.shape
        (OD2,L2) = prior_phase.shape
        (N,ID2) = x.shape
        (S,OD3,L3) = prior_weights.shape
        (OD4,L4) = outer_weights.shape
        (ID3,) = inner_weights.shape
        assert L==L2==L3==L4
        assert ID==ID2==ID3
        assert OD==OD2==OD3==OD4
        assert basis_weight.shape == (S,OD,L)
        assert basis_fn.shape == (OD,N,L)
        assert basis_fn_inner_prod.shape == (OD,N,L)
        assert rescaled_x.shape == (N,ID)
        assert output.shape == (S,OD,N)

        return output


    def prior_KL(
            self,
    ) -> jnp.ndarray:
        """Evaluates the prior KL term in the sparse VI objective. 
        
        """
        (OD,ID,M) = (self.output_dimension, self.input_dimension, self.num_inducing)
        inducing_locations = hk.get_parameter("inducing_locations", [M,ID], init=hk.initializers.RandomUniform())
        inducing_pseudo_mean = hk.get_parameter("inducing_pseudo_mean", [OD,M], init=jnp.zeros)
        inducing_pseudo_log_errvar = hk.get_parameter("inducing_pseudo_log_errvar", [OD,M], init=jnp.zeros)
        cholesky = hk.get_state("cholesky", [OD,M,M], init=jnp.zeros)
        
        logdet_term = (2 * jnp.sum(jax.vmap(jnp.diag)(cholesky))) - jnp.sum(inducing_pseudo_log_errvar)
        kernel_matrix = self.kernel.matrix(inducing_locations, inducing_locations)
        cholesky_inv = tf2jax.linalg.LinearOperatorLowerTriangular(cholesky).solve(jnp.eye(M))
        trace_term = jnp.sum(cholesky_inv * kernel_matrix)
        reparameterized_quadratic_form_term = jnp.sum(inducing_pseudo_mean @ kernel_matrix @ inducing_pseudo_mean.T)
        return (logdet_term - (OD*ID*M) + trace_term + reparameterized_quadratic_form_term) / 2

from typing import Optional
import haiku as hk
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import jax
import tensorflow_probability
tfp = tensorflow_probability.experimental.substrates.jax
tfk = tfp.math.psd_kernels
from sparsegpax.spectral import standard_spectral_measure, spectral_weights

class SparseGaussianProcess(): 
    """A sparse Gaussian process, implemented as a Haiku module.

    """


    def __init__(
            self,
            input_dimension: int,
            output_dimension: int,
            num_inducing: int,
            kernel: tfk.PositiveSemidefiniteKernel,
            key: jr.PRNGKey,
            name: Optional[str] = None,
    ):
        """Initializes the sparse GP.

        Args:
            name: module name.
        """
        # super().__init__(name=name)
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.num_inducing = num_inducing
        self.num_basis = 64
        self.num_samples = 8
        self.kernel = kernel

        (L,ID,OD,M,S) = (self.num_basis ,self.input_dimension, self.output_dimension, self.num_inducing, self.num_samples)

        self.prior_frequency = jnp.zeros((OD, ID, L))
        self.prior_phase = jnp.zeros((OD, L))
        self.prior_weights = jnp.zeros((OD, L, S))
        self.inducing_locations = jr.normal(key, (M, ID))
        self.inducing_pseudo_mean = jnp.zeros((OD, M))
        self.inducing_pseudo_log_errvar = jnp.ones((OD, M))
        self.inducing_weights = jnp.zeros((OD, M, S))
        self.cholesky = jnp.zeros((OD, M, M))
        
        self.resample_prior_basis(key)


    def __call__(
            self,
            x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluates the sparse GP for a given input matrix.

        Args:
            x: the input matrix.
        """
        (ID,OD,M,S) = (self.input_dimension, self.output_dimension, self.num_inducing, self.num_samples)
        # inducing_locations = hk.get_parameter("inducing_locations", [M,ID], init=hk.initializers.RandomUniform())
        # inducing_weights = hk.get_state("inducing_weights", [OD,M,S], init=jnp.zeros)
        inducing_locations = self.inducing_locations
        inducing_weights = self.inducing_weights

        f_prior = self.prior(x)
        f_data = (self.kernel.matrix(x, inducing_locations) @ inducing_weights).T # non-batched

        (N,ID) = x.shape
        (OD,M,S) = inducing_weights.shape
        assert f_prior.shape == (S,N,OD)
        assert f_data.shape == (S,N,OD)

        return f_prior + f_data


    def randomize(
            self,
            key: jr.PRNGKey,
            num_samples: int = None,
    ):
        """Samples a new set of random functions from the GP.

        Args:
            num_samples: the number of samples to draw.
            key: the random number generator key.
        """
        if num_samples is not None:
            self.num_samples = num_samples

        (S,M,OD,ID,L) = (self.num_samples, self.num_inducing, self.output_dimension, self.input_dimension, self.num_basis)
        # inducing_locations = hk.get_parameter("inducing_locations", [M,ID], init=hk.initializers.RandomUniform())
        # inducing_pseudo_mean = hk.get_parameter("inducing_pseudo_mean", [OD,M], init=jnp.zeros)
        # inducing_pseudo_log_errvar = hk.get_parameter("inducing_pseudo_log_errvar", [OD,M], init=jnp.zeros)
        inducing_locations = self.inducing_locations
        inducing_pseudo_mean = self.inducing_pseudo_mean
        inducing_pseudo_log_errvar = self.inducing_pseudo_log_errvar

        prior_weights = jr.normal(key, (OD,L,S))
        # hk.set_state("prior_weights", prior_weights)
        self.prior_weights = prior_weights

        (cholesky,_) = jsp.linalg.cho_factor(self.kernel.matrix(inducing_locations, inducing_locations) + jax.vmap(jnp.diag)(jnp.exp(inducing_pseudo_log_errvar)))
        residual = self.prior(inducing_locations).T - (jnp.reshape(jnp.exp(inducing_pseudo_log_errvar / 2), (OD,M,1)) * jr.normal(key,(OD,M,S))) # TODO: careful with f32!
        inducing_weights = jnp.reshape(inducing_pseudo_mean,(OD,M,1)) + jsp.linalg.cho_solve((cholesky,False),residual) # mean-reparameterized v = \mu + (K + V)^{-1}(f - \eps)

        # hk.set_state("inducing_weights", inducing_weights)
        # hk.set_state("cholesky", cholesky)
        self.inducing_weights = inducing_weights
        self.cholesky = cholesky
        
        assert prior_weights.shape == (OD,L,S)
        (M,ID) = inducing_locations.shape
        (OD2,M2,S) = self.inducing_weights.shape
        assert M==M2
        assert OD==OD2
        assert cholesky.shape==(OD,M,M)
        assert self.prior(inducing_locations).shape==(S,M,OD)
        assert residual.shape==(OD,M,S)
        assert inducing_weights.shape==(OD,M,S)


    def resample_prior_basis(
            self,
            key: jr.PRNGKey,
            num_basis: int = None
    ):
        """Resamples the frequency and phase of the prior random feature basis.

        Args:
            num_basis: the number of basis functions to use.
            key: the random number generator key.
        """
        if num_basis is not None:
            self.num_basis = num_basis

        (OD,ID,L) = (self.output_dimension, self.input_dimension, self.num_basis)
        prior_frequency = standard_spectral_measure(self.kernel, OD, ID, L, key)
        prior_phase = jr.uniform(key, (OD, L), maxval=2*jnp.pi)

        # hk.set_state("prior_frequency", prior_frequency)
        # hk.set_state("prior_phase", prior_phase)
        self.prior_frequency = prior_frequency
        self.prior_phase = prior_phase

        self.randomize(key)

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
        # prior_frequency = hk.get_state("prior_frequency", [OD,ID,L], init=jnp.zeros)
        # prior_phase = hk.get_state("prior_phase", [OD,L], init=jnp.zeros)
        # prior_weights = hk.get_state("prior_weights", [OD,L,S], init=jnp.zeros)
        prior_frequency = self.prior_frequency
        prior_phase = self.prior_phase
        prior_weights = self.prior_weights

        (outer_weights, inner_weights) = spectral_weights(self.kernel, prior_frequency)
        rescaled_x = x / inner_weights
        basis_fn_inner_prod = rescaled_x @ prior_frequency
        basis_fn = jnp.cos(basis_fn_inner_prod + jnp.reshape(prior_phase, (OD,1,L)))
        basis_weight = jnp.sqrt(2/L) * jnp.reshape(outer_weights,(OD,L,1)) * prior_weights # TODO: typecast this to f32!
        output = (basis_fn @ basis_weight).T

        (OD,ID,L) = prior_frequency.shape
        (OD2,L2) = prior_phase.shape
        (N,ID2) = x.shape
        (OD3,L3,S) = prior_weights.shape
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
        assert basis_weight.shape == (OD,L,S)
        assert basis_fn.shape == (OD,N,L)
        assert basis_fn_inner_prod.shape == (OD,N,L)
        assert rescaled_x.shape == (N,ID)
        assert output.shape == (S,N,OD)

        return output


    def prior_KL(
            self,
    ) -> jnp.ndarray:
        """Evaluates the prior KL term in the sparse VI objective. 
        
        """
        (OD,ID,M) = (self.output_dimension, self.input_dimension, self.num_inducing)
        # inducing_locations = hk.get_parameter("inducing_locations", [M,ID], init=hk.initializers.RandomUniform())
        # inducing_pseudo_mean = hk.get_parameter("inducing_pseudo_mean", [OD,M], init=jnp.zeros)
        # inducing_pseudo_log_errvar = hk.get_parameter("inducing_pseudo_log_errvar", [OD,M], init=jnp.zeros)
        # cholesky = hk.get_state("cholesky", [OD,M,M], init=jnp.zeros)
        inducing_locations = self.inducing_locations
        inducing_pseudo_mean = self.inducing_pseudo_mean
        inducing_pseudo_log_errvar = self.inducing_pseudo_log_errvar
        cholesky = self.cholesky

        logdet_term = (2 * jnp.sum(jax.vmap(jnp.diag)(cholesky))) - jnp.sum(inducing_pseudo_log_errvar)
        kernel_matrix = self.kernel.matrix(inducing_locations, inducing_locations)
        cholesky_inv = jax.vmap(lambda x: jsp.linalg.cho_solve((x,False), jnp.eye(*x.shape)))(cholesky)
        trace_term = jnp.sum(cholesky_inv * kernel_matrix)
        reparameterized_quadratic_form_term = jnp.sum(inducing_pseudo_mean @ kernel_matrix @ inducing_pseudo_mean.T)
        return (logdet_term - (OD*ID*M) + trace_term + reparameterized_quadratic_form_term) / 2

from typing import Optional
import haiku as hk
import jax
import jax.numpy as jnp

class SparseGaussianProcess(hk.Module): 
    def __init__(
        self,
        name: Optional[str] = None,
    ):
        """Initializes the sparse GP.

        Args:
            name: module name.
        """
        super().__init__(name=name)
        self.prior_frequency
        self.prior_phase
        self.prior_weights
        self.inducing_location
        self.inducing_pseudo_mean
        self.inducing_pseudo_noise
        self.inducing_weights
        self.cholesky_cache
        self.log_error
        self.hyperprior


    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluates the sparse GP for a given input matrix.

        Args:
            x: the input matrix.
        """

    def randomize(
        self
    ):
        """Samples a new set of random functions from the GP.

        """

    def resample_prior_basis(
        self
    ):
        """Resamples the frequency and phase of the prior random feature basis.

        """

    def eval_prior(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluates the prior GP at x.
        
        Args:
            x: the input matrix.
        """

    def prior_KL(
        self
    ) -> jnp.ndarray:
        """Evaluates the prior KL term in the sparse VI objective.

        """
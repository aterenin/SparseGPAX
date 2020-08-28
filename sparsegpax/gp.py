from typing import Optional, NamedTuple
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

class SparseGaussianProcessState(NamedTuple): 
    """Due to Haiku limitation we need to manually manage state.

    """
    prior_frequency: jnp.ndarray
    prior_phase: jnp.ndarray
    prior_weights: jnp.ndarray
    inducing_weights: jnp.ndarray
    cholesky: jnp.ndarray

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

        (S,OD,ID,M,L) = (self.num_samples, self.output_dimension, self.input_dimension, self.num_inducing, self.num_basis)
        hk.get_parameter("log_error_stddev", [OD], init=jnp.zeros)
        hk.get_parameter("inducing_locations", [M,ID], init=hk.initializers.RandomUniform())
        hk.get_parameter("inducing_pseudo_mean", [OD,M], init=jnp.zeros)
        hk.get_parameter("inducing_pseudo_log_err_stddev", [OD,M], init=jnp.zeros)


    def get_initial_state(
            self
    ) -> SparseGaussianProcessState:
        (S,OD,ID,M,L) = (self.num_samples, self.output_dimension, self.input_dimension, self.num_inducing, self.num_basis)
        prior_frequency = jnp.zeros((OD,ID,L))
        prior_phase = jnp.zeros((OD,L))
        prior_weights = jnp.zeros((S,OD,L))
        inducing_weights = jnp.zeros((S,OD,M))
        cholesky = jnp.zeros((OD,M,M))
        return SparseGaussianProcessState(
            prior_frequency,
            prior_phase,
            prior_weights,
            inducing_weights,
            cholesky
            )


    def __call__(
            self,
            x: jnp.ndarray,
            state: SparseGaussianProcessState
    ) -> jnp.ndarray:
        """Evaluates the sparse GP for a given input matrix.

        Args:
            x: the input matrix.
        """
        (S,OD,ID,M) = (self.num_samples, self.output_dimension, self.input_dimension, self.num_inducing)
        inducing_locations = hk.get_parameter("inducing_locations", [M,ID], init=hk.initializers.RandomUniform())
        inducing_weights = state.inducing_weights
        
        f_prior = self.prior(x,state)
        f_data = tf2jax.linalg.matvec(self.kernel.matrix(x, inducing_locations), inducing_weights) # non-batched

        (N,ID) = x.shape
        (S,OD,M) = inducing_weights.shape
        assert f_prior.shape == (S,OD,N)
        assert f_data.shape == (S,OD,N)

        return f_prior + f_data


    def randomize(
            self,
            state: SparseGaussianProcessState
    ) -> SparseGaussianProcessState:
        """Samples a new set of random functions from the GP.

        """
        epsilon_fixed = jnp.expand_dims(jnp.array([[0.101628,0.746622,1.35145,0.356882,-0.213243,-0.0334132,0.300289,0.226092,-1.5504,2.36453,0.817478,-0.177483,-0.545376,-0.429228,-0.324968,-1.65114,-1.29277],[-1.66014,-2.33706,0.366218,1.3569,-2.46641,-0.0488879,-2.20564,-1.13134,0.383122,0.15138,-1.63817,-0.846391,-0.64897,0.959584,-1.53584,-0.62623,-1.11368],[-0.594149,1.75836,0.0374476,-1.45139,0.811625,1.68822,-2.0736,-0.263524,0.268272,-0.576364,-0.0453364,-2.07196,-0.53978,1.42136,-1.22146,1.57069,1.13494],[0.690387,-0.274249,-0.213479,2.3303,-2.6079,-0.478749,0.518475,-0.964862,-0.0531467,0.773948,0.529792,-0.08998,0.441066,0.200661,0.501974,-1.28511,0.723244],[0.203887,-2.67418,-0.510925,-0.0745303,1.96131,-0.0408828,-1.06171,0.268955,-0.221138,-0.898042,-0.304627,0.83713,-1.41777,2.43939,2.10231,0.782125,0.563489],[-1.37911,-1.32241,-1.54064,0.545873,0.781736,-0.867173,-0.655529,-1.2211,-0.250992,0.460903,0.657106,0.304237,1.50716,0.695404,-1.05009,-0.0437475,0.0943785],[1.21814,0.00312229,-0.551767,-0.588853,-2.54814,-0.274372,-0.850338,0.336523,1.32281,1.63375,-0.732341,0.313368,0.864539,-0.0969507,0.916594,0.00308649,1.108],[-1.75284,-0.554282,-0.707712,0.681605,-1.16549,0.736054,0.316533,-0.449797,1.36436,-2.14107,0.213541,1.57444,-0.00435671,0.127655,-0.18457,0.0510322,-0.292553],[-0.552427,0.797645,-0.465199,-0.176894,-0.546766,-0.998528,1.3059,-0.475434,1.34267,0.578365,1.44085,-1.0061,1.00931,-0.455156,-0.097822,0.820066,-0.552937],[-0.599359,-1.83851,-0.329292,-1.11202,0.678707,0.504035,0.165718,-0.0490024,-1.54964,1.29132,1.07487,-0.0640134,0.304575,-1.51363,1.06762,-0.0403226,0.643361],[1.94725,1.88345,0.207108,2.58147,2.14006,-0.18929,-0.724996,0.000225098,0.998617,-0.0225029,-0.36584,0.638425,-1.30143,0.364434,-0.00522458,-1.16472,0.972106]]).T,1)

        (S,OD,ID,M,L) = (self.num_samples, self.output_dimension, self.input_dimension, self.num_inducing, self.num_basis)
        inducing_locations = hk.get_parameter("inducing_locations", [M,ID], init=hk.initializers.RandomUniform())
        inducing_pseudo_mean = hk.get_parameter("inducing_pseudo_mean", [OD,M], init=jnp.zeros)
        inducing_pseudo_log_err_stddev = hk.get_parameter("inducing_pseudo_log_err_stddev", [OD,M], init=jnp.zeros)
        
        prior_weights = jr.normal(hk.next_rng_key(), (S,OD,L))
        state = SparseGaussianProcessState(state.prior_frequency, state.prior_phase, prior_weights, state.inducing_weights, state.cholesky)
        
        (cholesky,_) = jsp.linalg.cho_factor(self.kernel.matrix(inducing_locations, inducing_locations) + jax.vmap(jnp.diag)(jnp.exp(inducing_pseudo_log_err_stddev * 2)), lower=True)
        residual = self.prior(inducing_locations,state) + jnp.exp(inducing_pseudo_log_err_stddev) * jr.normal(hk.next_rng_key(),(S,OD,M))
        inducing_weights = inducing_pseudo_mean - tf2jax.linalg.LinearOperatorLowerTriangular(cholesky).solvevec(tf2jax.linalg.LinearOperatorLowerTriangular(cholesky).solvevec(residual), adjoint=True) # mean-reparameterized v = \mu + (K + V)^{-1}(-f - \eps)  
        
        assert prior_weights.shape == (S,OD,L)
        (M,ID) = inducing_locations.shape
        (S,OD2,M2) = inducing_weights.shape
        assert M==M2
        assert OD==OD2
        assert cholesky.shape==(OD,M,M)
        assert self.prior(inducing_locations,state).shape==(S,OD,M)
        assert residual.shape==(S,OD,M)
        assert inducing_weights.shape==(S,OD,M)

        return SparseGaussianProcessState(
            state.prior_frequency,
            state.prior_phase,
            prior_weights,
            inducing_weights,
            cholesky
            )


    def resample_prior_basis(
            self,
            state: SparseGaussianProcessState,
    ):
        """Resamples the frequency and phase of the prior random feature basis.

        """
        (OD,ID,L) = (self.output_dimension, self.input_dimension, self.num_basis)
        prior_frequency = standard_spectral_measure(self.kernel, OD, ID, L, hk.next_rng_key())
        prior_phase = jr.uniform(hk.next_rng_key(), (OD, L), maxval=2*jnp.pi)

        (OD,ID,L) = prior_frequency.shape
        assert prior_frequency.shape == (OD,ID,L)
        (OD2,L2) = prior_phase.shape
        assert OD2==OD
        assert L2==L
        assert prior_phase.shape == (OD,L)

        return SparseGaussianProcessState(
            prior_frequency,
            prior_phase,
            state.prior_weights,
            state.inducing_weights,
            state.cholesky
            )


    def prior(
            self,
            x: jnp.ndarray,
            state: SparseGaussianProcessState,
    ) -> jnp.ndarray:
        """Evaluates the prior GP at x.

        Args:
            x: the input matrix.
        """
        (S,OD,ID,L) = (self.num_samples, self.output_dimension, self.input_dimension, self.num_basis)
        prior_frequency = state.prior_frequency
        prior_phase = state.prior_phase
        prior_weights = state.prior_weights
        
        (outer_weights, inner_weights) = spectral_weights(self.kernel, prior_frequency)
        rescaled_x = x / inner_weights
        basis_fn_inner_prod = rescaled_x @ prior_frequency
        basis_fn = jnp.cos(basis_fn_inner_prod + jnp.expand_dims(prior_phase,-2))
        basis_weights = jnp.sqrt(2/L) * outer_weights * prior_weights
        output = tf2jax.linalg.matvec(basis_fn, basis_weights)

        (OD,ID,L) = prior_frequency.shape
        (OD2,L2) = prior_phase.shape
        (N,ID2) = x.shape
        (S,OD3,L3) = prior_weights.shape
        (OD4,L4) = outer_weights.shape
        (ID3,) = inner_weights.shape
        assert L==L2==L3==L4
        assert ID==ID2==ID3
        assert OD==OD2==OD3==OD4
        assert basis_weights.shape == (S,OD,L)
        assert basis_fn.shape == (OD,N,L)
        assert basis_fn_inner_prod.shape == (OD,N,L)
        assert rescaled_x.shape == (N,ID)
        assert output.shape == (S,OD,N)

        return output


    def prior_KL(
            self,
            state: SparseGaussianProcessState,
    ) -> jnp.ndarray:
        """Evaluates the prior KL term in the sparse VI objective. 
        
        """
        (OD,ID,M) = (self.output_dimension, self.input_dimension, self.num_inducing)
        inducing_locations = hk.get_parameter("inducing_locations", [M,ID], init=hk.initializers.RandomUniform())
        inducing_pseudo_mean = hk.get_parameter("inducing_pseudo_mean", [OD,M], init=jnp.zeros)
        inducing_pseudo_log_err_stddev = hk.get_parameter("inducing_pseudo_log_err_stddev", [OD,M], init=jnp.zeros)
        cholesky = state.cholesky
        
        logdet_term = 2*jnp.sum(jnp.log(jax.vmap(jnp.diag)(cholesky))) - 2*jnp.sum(inducing_pseudo_log_err_stddev)
        kernel_matrix = self.kernel.matrix(inducing_locations, inducing_locations)
        cholesky_inv = tf2jax.linalg.LinearOperatorLowerTriangular(cholesky).solve(tf2jax.linalg.LinearOperatorLowerTriangular(cholesky).solve(jnp.eye(M)),adjoint=True)
        trace_term = jnp.sum(cholesky_inv * kernel_matrix)
        reparameterized_quadratic_form_term = jnp.sum(inducing_pseudo_mean * tf2jax.linalg.matvec(kernel_matrix, inducing_pseudo_mean))
        return (logdet_term - (OD*ID*M) + trace_term + reparameterized_quadratic_form_term) / 2

    def err_stddev(
            self,
    ) -> jnp.ndarray:
        """Returns the error variance vector of the GP.
        
        """
        return jnp.exp(hk.get_parameter("log_error_stddev", [self.output_dimension], init=jnp.zeros))

    def hyperprior(
            self,
    ) -> jnp.ndarray:
        """Returns the log hyperprior regularization term of the GP.
        
        """
        return jnp.zeros(()) # temporary
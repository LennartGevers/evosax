"""Adaptive ES-Active Subspaces for Blackbox Optimization (ASEBO).

The implementation closely follows Choromanski et al. (2019)
``https://arxiv.org/abs/1903.04268`` while adapting the algorithm to the
batch-oriented interface used throughout :mod:`evosax`.  Notable
differences from the original presentation include:

* Always sampling a fixed population size per generation to simplify JIT
  compilation and batching.
* Maintaining a fixed-size FIFO archive of gradient estimates when
  constructing the low-dimensional subspace.

These docstrings intentionally detail the behaviour so that the MkDocs API
reference generated via mkdocstrings communicates the assumptions and
operational nuances to practitioners exploring the library.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax import struct

from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution

from .base import (
    DistributionBasedAlgorithm,
    Params as BaseParams,
    State as BaseState,
    metrics_fn,
)


@struct.dataclass
class State(BaseState):
    """Additional state tracked by :class:`ASEBO`.

    Attributes
    ----------
    mean:
        Flattened mean of the search distribution (inherited from
        :class:`DistributionBasedAlgorithm`).
    std:
        Standard deviation used when sampling the isotropic exploration
        component.
    opt_state:
        Internal optimiser state returned by the Optax transformation that
        updates the mean.
    grad_subspace:
        FIFO buffer of recent gradient estimates used to construct the
        active subspace via SVD.
    alpha:
        Mixing coefficient that balances the isotropic and subspace-based
        covariance components.
    UUT:
        Covariance contribution formed by projecting onto the active
        subspace.
    UUT_ort:
        Covariance contribution formed by projecting onto the orthogonal
        complement of the active subspace.
    """

    mean: jax.Array
    std: float
    opt_state: optax.OptState
    grad_subspace: jax.Array
    alpha: float
    UUT: jax.Array
    UUT_ort: jax.Array


@struct.dataclass
class Params(BaseParams):
    """Hyper-parameters specific to :class:`ASEBO`.

    Attributes
    ----------
    grad_decay:
        Exponential decay used when maintaining the FIFO gradient archive.
        Values close to ``1.0`` retain gradients for longer, whereas lower
        values make the subspace react faster to new information.
    """

    grad_decay: float


class ASEBO(DistributionBasedAlgorithm):
    """Distribution-based algorithm that adapts an active subspace online.

    ASEBO balances exploration between the global search space and a
    low-dimensional active subspace estimated from historical gradients.
    This yields gradient-efficient search directions while retaining enough
    isotropic exploration to escape poor subspaces.
    """

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        subspace_dims: int = 1,
        optimizer: optax.GradientTransformation = optax.adam(learning_rate=1e-3),
        std_schedule: Callable = optax.constant_schedule(1.0),
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Construct an ASEBO optimiser instance.

        Parameters
        ----------
        population_size:
            Number of individuals sampled each generation.  Must be even to
            support antithetic sampling.
        solution:
            Representative solution PyTree whose structure defines the
            optimisation space.
        subspace_dims:
            Number of principal directions retained when constructing the
            active subspace.  Defaults to ``1``.
        optimizer:
            Optax gradient transformation responsible for updating the mean.
        std_schedule:
            Callable returning the standard deviation used for sampling at a
            given generation counter.
        fitness_shaping_fn:
            Callable that reshapes raw fitness values prior to gradient
            estimation.  Uses :func:`identity_fitness_shaping_fn` by default.
        metrics_fn:
            Callable producing diagnostic metrics after each ``tell`` step.
        """
        assert population_size % 2 == 0, "Population size must be even."
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        assert subspace_dims <= self.num_dims, (
            "Subspace dims must be smaller than optimization dims."
        )
        self.subspace_dims = subspace_dims

        # Optimizer
        self.optimizer = optimizer

        # std schedule
        self.std_schedule = std_schedule

    @property
    def _default_params(self) -> Params:
        """Return algorithm defaults used when no parameters are supplied."""
        return Params(grad_decay=0.99)

    def _init(self, key: jax.Array, params: Params) -> State:
        """Initialise the ASEBO state prior to the first optimisation step."""
        grad_subspace = jnp.zeros((self.subspace_dims, self.num_dims))

        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=self.std_schedule(0),
            opt_state=self.optimizer.init(jnp.zeros(self.num_dims)),
            grad_subspace=grad_subspace,
            alpha=1.0,
            UUT=jnp.zeros((self.num_dims, self.num_dims)),
            UUT_ort=jnp.zeros((self.num_dims, self.num_dims)),
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
        )
        return state

    def _ask(
        self,
        key: jax.Array,
        state: State,
        params: Params,
    ) -> tuple[Population, State]:
        """Sample a new population of candidates from the current search distribution.

        The method implements antithetic sampling around the current mean
        while mixing isotropic noise with directions derived from the active
        subspace.  It also updates cached covariance terms used during the
        subsequent :meth:`_tell` call.
        """
        # Antithetic sampling of noise
        X = state.grad_subspace
        X -= jnp.mean(X, axis=0)
        U, S, Vt = jnp.linalg.svd(X, full_matrices=False)

        def svd_flip(u, v):
            # columns of u, rows of v
            max_abs_cols = jnp.argmax(jnp.abs(u), axis=0)
            signs = jnp.sign(u[max_abs_cols, jnp.arange(u.shape[1])])
            u *= signs
            v *= signs[:, jnp.newaxis]
            return u, v

        U, Vt = svd_flip(U, Vt)
        U = Vt[: int(self.population_size / 2)]
        UUT = jnp.matmul(U.T, U)

        U_ort = Vt[int(self.population_size / 2) :]
        UUT_ort = jnp.matmul(U_ort.T, U_ort)

        subspace_ready = state.generation_counter > self.subspace_dims

        UUT = jax.lax.select(
            subspace_ready, UUT, jnp.zeros((self.num_dims, self.num_dims))
        )
        cov = (
            state.std * (state.alpha / self.num_dims) * jnp.eye(self.num_dims)
            + ((1 - state.alpha) / int(self.population_size / 2)) * UUT
        )
        chol = jnp.linalg.cholesky(cov)
        z_plus = jax.random.normal(key, (self.population_size // 2, self.num_dims))
        z_plus = z_plus @ chol.T
        z_plus /= jnp.linalg.norm(z_plus, axis=-1)[:, None]
        z = jnp.concatenate([z_plus, -z_plus])
        population = state.mean + z
        return population, state.replace(UUT=UUT, UUT_ort=UUT_ort)

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        """Update the internal state using evaluated fitness values.

        The fitness differences between antithetic pairs are converted into
        a gradient estimate that updates the mean via the configured Optax
        optimiser.  The gradient archive and subspace mixing coefficient are
        refreshed to prepare for the next call to :meth:`_ask`.
        """
        # Compute grad
        fitness_plus = fitness[: self.population_size // 2]
        fitness_minus = fitness[self.population_size // 2 :]
        grad = 0.5 * jnp.dot(
            fitness_plus - fitness_minus,
            (population[: self.population_size // 2] - state.mean) / state.std,
        )

        alpha = jnp.linalg.norm(jnp.dot(grad, state.UUT_ort)) / jnp.linalg.norm(
            jnp.dot(grad, state.UUT)
        )
        subspace_ready = state.generation_counter > self.subspace_dims
        alpha = jax.lax.select(subspace_ready, alpha, 1.0)

        # FIFO grad subspace (same as in guided_es.py)
        grad_subspace = jnp.roll(state.grad_subspace, shift=-1, axis=0)
        grad_subspace = grad_subspace.at[-1, :].set(grad)

        # Normalize gradients by norm / num_dims
        grad /= jnp.linalg.norm(grad) / self.num_dims + 1e-8

        # Update mean
        updates, opt_state = self.optimizer.update(grad, state.opt_state)
        mean = optax.apply_updates(state.mean, updates)

        return state.replace(
            mean=mean,
            std=self.std_schedule(state.generation_counter),
            opt_state=opt_state,
            grad_subspace=grad_subspace,
            alpha=alpha,
        )

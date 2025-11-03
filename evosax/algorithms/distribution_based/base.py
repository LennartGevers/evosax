"""Shared utilities for distribution-based evolutionary strategies.

This module provides the light-weight scaffolding that concrete
distribution-based optimizers, such as ASEBO or SNES, build upon.  The
shared `State` and `Params` dataclasses extend the generic evolutionary
algorithm containers with a `mean` vector representing the centre of the
search distribution.  Additionally the module offers a convenience
``metrics_fn`` that enriches the base metrics with diagnostics that are
particularly relevant for this family of algorithms (e.g. the norm of the
distribution mean).

While simple, documenting the behaviour here helps the generated API
reference explain how distribution-based algorithms interact with the rest
of :mod:`evosax`.  These docstrings are intentionally verbose so that the
rendered documentation can guide users exploring the codebase via
mkdocstrings.
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Fitness, Metrics, Population, Solution

from ..base import (
    EvolutionaryAlgorithm,
    Params as BaseParams,
    State as BaseState,
    metrics_fn as base_metrics_fn,
)


@struct.dataclass
class State(BaseState):
    """State container shared by distribution-based strategies.

    Attributes
    ----------
    mean:
        The flattened mean solution representing the centre of the
        sampling distribution used to generate candidate populations.  It
        is stored in the flattened space produced by
        :func:`jax.flatten_util.ravel_pytree` to avoid repeatedly
        reshaping PyTrees during inner-loop updates.
    """

    mean: Solution


@struct.dataclass
class Params(BaseParams):
    """Parameter bundle for distribution-based algorithms.

    This class does not introduce additional hyper-parameters beyond the
    shared :class:`~evosax.algorithms.base.Params`.  The dedicated subclass
    is retained to provide a stable extension point – concrete algorithms
    may override the default parameters with distribution-specific fields
    (e.g. ``grad_decay`` for ASEBO).
    """


def metrics_fn(
    key: jax.Array,
    population: Population,
    fitness: Fitness,
    state: State,
    params: Params,
) -> Metrics:
    """Augment the base metrics with distribution diagnostics.

    The base :func:`~evosax.algorithms.base.metrics_fn` tracks scalar
    statistics such as best and average fitness.  Distribution-based
    algorithms benefit from monitoring the mean vector as well, therefore
    this helper adds the mean itself and its :math:`L_2` norm to the
    metrics dictionary returned by the algorithm.

    Parameters
    ----------
    key:
        PRNG key forwarded to the base metrics function for deterministic
        aggregations.
    population:
        Evaluated population of candidate solutions.
    fitness:
        Fitness scores corresponding to ``population``.
    state:
        Current algorithm state produced by
        :class:`DistributionBasedAlgorithm`.
    params:
        Algorithm hyper-parameters.

    Returns
    -------
    dict[str, jax.Array | float]
        The combined metrics containing both base statistics and the mean
        diagnostics required by distribution-based strategies.
    """
    metrics = base_metrics_fn(key, population, fitness, state, params)
    return metrics | {
        "mean": state.mean,
        "mean_norm": jnp.linalg.norm(state.mean, axis=-1),
    }


class DistributionBasedAlgorithm(EvolutionaryAlgorithm):
    """Base class for distribution-based evolutionary algorithms.

    Concrete subclasses model their search distribution via a mean vector
    and potentially additional distribution parameters (such as a
    covariance matrix).  This helper wires the shared initialisation logic
    and exposes convenience methods for working with the mean in flattened
    and unflattened forms.
    """

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Configure the shared components of a distribution strategy.

        Parameters
        ----------
        population_size:
            Number of candidate solutions sampled per generation.
        solution:
            PyTree describing the shape of a single solution; used for
            flattening/unflattening mean vectors.
        fitness_shaping_fn:
            Callable applied to raw fitness scores prior to computing
            gradients.  Defaults to :func:`identity_fitness_shaping_fn`.
        metrics_fn:
            Callable used to collect metrics after each ``tell`` step.  By
            default this module's :func:`metrics_fn` is used, adding mean
            diagnostics to the base metrics.
        """
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        key: jax.Array,
        mean: Solution,
        params: Params,
    ) -> State:
        """Create a freshly initialised algorithm state.

        Parameters
        ----------
        key:
            Random key that seeds the state initialisation.
        mean:
            Initial mean solution around which candidate populations will
            be sampled.
        params:
            Hyper-parameters controlling the algorithm behaviour.

        Returns
        -------
        State
            The populated state dataclass with the provided mean encoded in
            flattened form.
        """
        state = self._init(key, params)
        state = state.replace(mean=self._ravel_solution(mean))
        return state

    def get_mean(self, state: State) -> Solution:
        """Return the mean solution in its original PyTree structure."""
        mean = self._unravel_solution(state.mean)
        return mean

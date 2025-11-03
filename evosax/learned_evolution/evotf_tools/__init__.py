"""Utilities shared across Evolution Transformer components.

The submodules in this package have substantial import-time dependencies (for
example, `features` reaches into the algorithm registry). To avoid circular
imports, we expose the public API via lazy attribute access.
"""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "EvoTransformer",
    "FitnessFeaturizer",
    "FitnessFeaturesState",
    "SolutionFeaturizer",
    "SolutionFeaturesState",
    "DistributionFeaturizer",
    "DistributionFeaturesState",
]

_ATTRIBUTE_MODULE_MAP = {
    "EvoTransformer": "evosax.learned_evolution.evotf_tools.evo_transformer",
    "FitnessFeaturizer": "evosax.learned_evolution.evotf_tools.features.fitness",
    "FitnessFeaturesState": "evosax.learned_evolution.evotf_tools.features.fitness",
    "SolutionFeaturizer": "evosax.learned_evolution.evotf_tools.features.solution",
    "SolutionFeaturesState": "evosax.learned_evolution.evotf_tools.features.solution",
    "DistributionFeaturizer": "evosax.learned_evolution.evotf_tools.features.distribution",
    "DistributionFeaturesState": "evosax.learned_evolution.evotf_tools.features.distribution",
}


def __getattr__(name):
    try:
        module_name = _ATTRIBUTE_MODULE_MAP[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    return getattr(module, name)


def __dir__():
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover
    from .evo_transformer import EvoTransformer
    from .features.distribution import DistributionFeaturesState, DistributionFeaturizer
    from .features.fitness import FitnessFeaturesState, FitnessFeaturizer
    from .features.solution import SolutionFeaturesState, SolutionFeaturizer

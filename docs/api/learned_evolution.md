# Learned evolution API

Modules in `evosax.learned_evolution` contain learned controllers, score models, and utilities that extend classical evolution strategies with trainable components.

## Fitness shaping

::: evosax.learned_evolution.fitness_shaping
    options:
        members: true

## Learned ES utilities

::: evosax.learned_evolution.les_tools
    options:
        members: true

## Learned GA utilities

::: evosax.learned_evolution.lga_tools
    options:
        members: true

## Evo Transformer utilities

::: evosax.learned_evolution.evotf_tools
    options:
        members:
            - EvoTransformer
            - DistributionFeaturizer
            - DistributionFeaturesState
            - FitnessFeaturizer
            - FitnessFeaturesState
            - SolutionFeaturizer
            - SolutionFeaturesState

The full evolution strategy powered by these components is documented under [`EvoTF_ES`](algorithms.md#evosax.algorithms.distribution_based.evotf_es.EvoTF_ES).

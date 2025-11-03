# Population-based strategies

Population-based algorithms explicitly evolve a set of candidate solutions using operators such as selection, crossover, mutation, and recombination. `evosax` implements these methods with JAX-compatible state to support batching, vectorization, and accelerator execution.

## Genetic algorithms

- [`SimpleGA`](../api/algorithms.md#evosax.algorithms.population_based.simple_ga.SimpleGA) – a classic $(μ + λ)$ genetic algorithm with tournament selection and mutation.
- [`MR15_GA`](../api/algorithms.md#evosax.algorithms.population_based.mr15_ga.MR15_GA) – a multi-recombinative variant with adaptive mutation rates.
- [`SAMR_GA`](../api/algorithms.md#evosax.algorithms.population_based.samr_ga.SAMR_GA) and [`GESMR_GA`](../api/algorithms.md#evosax.algorithms.population_based.gesmr_ga.GESMR_GA) – self-adaptive mutation rate algorithms for robust exploration.
- [`LearnedGA`](../api/algorithms.md#evosax.algorithms.population_based.learned_ga.LearnedGA) – augments classic GA operators with learned mutation policies.

## Swarm and diffusion methods

- [`PSO`](../api/algorithms.md#evosax.algorithms.population_based.pso.PSO) – particle swarm optimization with inertia and social components.
- [`DiffusionEvolution`](../api/algorithms.md#evosax.algorithms.population_based.diffusion_evolution.DiffusionEvolution) – diffusion-based search that transitions populations using learned score networks.
- [`DifferentialEvolution`](../api/algorithms.md#evosax.algorithms.population_based.differential_evolution.DifferentialEvolution) – classic DE/rand/1 strategy with configurable crossover and scaling factors.

## Hybrid search

Combine population-based algorithms with [distribution-based strategies](distribution.md) using shared objective evaluations or ensemble approaches. Because all states are pytrees, you can compose them with JAX control flows or plug them into RL training loops.

## Example notebooks

| Notebook | Highlights |
| --- | --- |
| [01_bbob.ipynb](https://github.com/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb) | Benchmark multiple GA and DE variants on standard objective functions. |
| [05_diffusion_evolution.ipynb](https://github.com/RobertTLange/evosax/blob/main/examples/05_diffusion_evolution.ipynb) | Explore diffusion evolution on challenging optimization tasks. |
| [06_sv_es.ipynb](https://github.com/RobertTLange/evosax/blob/main/examples/06_sv_es.ipynb) | Compare Stein variational strategies with swarm-based baselines. |

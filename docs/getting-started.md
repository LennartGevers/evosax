# Getting started

This guide walks through installing `evosax`, running your first ask-eval-tell loop, and highlights the most common configuration options.

## Installation

`evosax` supports Python 3.10+ and requires a working [JAX](https://github.com/google/jax) installation. Install the latest release from PyPI:

```bash
pip install evosax
```

To try the bleeding-edge version with the newest algorithms and fixes, install directly from GitHub:

```bash
pip install "git+https://github.com/RobertTLange/evosax.git@main"
```

For GPU/TPU execution, follow the [official JAX installation instructions](https://jax.readthedocs.io/en/latest/installation.html) to pick the correct wheels before installing `evosax`.

## The ask-eval-tell loop

All strategies in `evosax` implement the same high-level workflow:

1. **Ask** – sample a batch of candidate solutions from the search distribution or population.
2. **Eval** – compute fitness values for each candidate in your objective function.
3. **Tell** – update the algorithm state using the observed fitness scores.

```python
import jax
from evosax.algorithms import CMA_ES

# 1) Configure the strategy
es = CMA_ES(population_size=32, solution=dummy_solution)
params = es.default_params

# 2) Initialize internal state
key = jax.random.key(0)
state = es.init(key, params)

# 3) Iterate the ask-eval-tell loop
for step in range(num_generations):
    key, key_ask, key_eval = jax.random.split(key, 3)
    population, state = es.ask(key_ask, state, params)
    fitness = objective(population)
    state = es.tell(population, fitness, state, params)
```

Strategies expose a `best_solution` and `best_fitness` via the returned state to monitor progress. Because everything is JAX-compatible, you can wrap the loop with `jax.jit` or vectorize across environments with `jax.vmap`.

## Choosing an algorithm

`evosax` ships with both distribution-based and population-based algorithms:

- **Distribution-based** methods such as [`CMA_ES`](algorithms/distribution.md#covariance-matrix-adaptation) and [`Open_ES`](api/algorithms.md#evosax.algorithms.Open_ES) maintain and adapt a sampling distribution.
- **Population-based** methods such as [`SimpleGA`](algorithms/population.md#genetic-algorithms) and [`DiffusionEvolution`](api/algorithms.md#evosax.algorithms.DiffusionEvolution) evolve a population of explicit candidate solutions.

Refer to the [algorithms overview](algorithms/distribution.md) for a curated list of strategies and example notebooks.

## Next steps

- Learn about restart schedules, fitness shaping, and optimizers in the [Core utilities](api/core.md) reference.
- Browse ready-to-run notebooks in the [`examples/`](https://github.com/RobertTLange/evosax/tree/main/examples) directory.
- Check out the [API reference](api/algorithms.md) for parameter defaults and state structure of each strategy.

# Ask-eval-tell workflow

Every algorithm in `evosax` implements the same functional interface built around `ask`, `tell`, and immutable state updates. This guide explains how the pieces fit together so you can integrate evolution strategies into larger JAX programs.

## Anatomy of an algorithm

Each strategy exposes three core attributes:

- **`default_params`** – a pytree with algorithm-specific hyperparameters.
- **`init(key, params)`** – creates the internal state, including the current mean, covariance, or population.
- **`ask(key, state, params)`** – samples candidate solutions and returns `(population, new_state)`.
- **`tell(population, fitness, state, params)`** – updates the state using evaluated fitness values.

The [`CMA_ES` implementation](../api/algorithms.md#evosax.algorithms.CMA_ES) demonstrates this interface:

```python
from evosax.algorithms import CMA_ES

es = CMA_ES(population_size=32, solution=dummy_solution)
params = es.default_params
state = es.init(key, params)
population, state = es.ask(key, state, params)
state = es.tell(population, fitness, state, params)
```

## Working with JAX transformations

Because algorithm state and parameters are pytrees, they work seamlessly with JAX transformations:

- **`jax.jit`** compiles the entire loop for accelerator execution.
- **`jax.vmap`** batches multiple objectives or environments.
- **`jax.lax.scan`** unrolls the ask-eval-tell iterations without Python overhead.

```python
import jax

@jax.jit
def optimize(key, params, num_generations):
    state = es.init(key, params)

    def step(carry, _):
        key, state = carry
        key, key_ask, key_eval = jax.random.split(key, 3)
        population, state = es.ask(key_ask, state, params)
        fitness = objective(population)
        state = es.tell(population, fitness, state, params)
        return (key, state), None

    (key, state), _ = jax.lax.scan(step, (key, state), None, length=num_generations)
    return state
```

## Handling randomness

Pass explicit PRNG keys into each `ask` call to keep randomness reproducible. Many algorithms accept structured keys (e.g. `(key, key_sigma)`) for deterministic evaluation. If you are running multiple replicas, split the global key for each worker and pass it down to maintain independence.

## Monitoring progress

The returned state typically includes:

- `best_fitness` and `best_solution`
- Running estimates such as covariance matrices, step sizes, or mutation rates
- Bookkeeping information (generation counters, restart status)

You can log these fields directly or transform them with `jax.device_get` when interacting with host-side logging frameworks.

For advanced instrumentation, inspect the dataclasses defined in [`evosax.types`](../api/types.md#evosax.types) to understand the full schema used by strategies.

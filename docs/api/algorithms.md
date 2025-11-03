# Algorithms API

`evosax.algorithms` exposes both distribution-based and population-based evolution strategies behind a unified ask-eval-tell interface. All classes are JAX pytrees with immutable state updates to ensure compatibility with `jax.jit`, `jax.vmap`, and `jax.lax.scan`.

## Base classes

::: evosax.algorithms.base.EvolutionaryAlgorithm

::: evosax.algorithms.distribution_based.base.DistributionBasedAlgorithm

::: evosax.algorithms.population_based.base.PopulationBasedAlgorithm

::: evosax.algorithms.distribution_based.sv.base.SV_ES

## Distribution-based strategies

::: evosax.algorithms.distribution_based.ars.ARS

::: evosax.algorithms.distribution_based.asebo.ASEBO

::: evosax.algorithms.distribution_based.cma_es.CMA_ES

::: evosax.algorithms.distribution_based.cr_fm_nes.CR_FM_NES

::: evosax.algorithms.distribution_based.discovered_es.DiscoveredES

::: evosax.algorithms.distribution_based.esmc.ESMC

::: evosax.algorithms.distribution_based.evotf_es.EvoTF_ES

::: evosax.algorithms.distribution_based.gradientless_descent.GradientlessDescent

::: evosax.algorithms.distribution_based.guided_es.GuidedES

::: evosax.algorithms.distribution_based.hill_climbing.HillClimbing

::: evosax.algorithms.distribution_based.iamalgam_full.iAMaLGaM_Full

::: evosax.algorithms.distribution_based.iamalgam_univariate.iAMaLGaM_Univariate

::: evosax.algorithms.distribution_based.learned_es.LearnedES

::: evosax.algorithms.distribution_based.lm_ma_es.LM_MA_ES

::: evosax.algorithms.distribution_based.ma_es.MA_ES

::: evosax.algorithms.distribution_based.noise_reuse_es.NoiseReuseES

::: evosax.algorithms.distribution_based.open_es.Open_ES

::: evosax.algorithms.distribution_based.persistent_es.PersistentES

::: evosax.algorithms.distribution_based.pgpe.PGPE

::: evosax.algorithms.distribution_based.random_search.RandomSearch

::: evosax.algorithms.distribution_based.rm_es.Rm_ES

::: evosax.algorithms.distribution_based.sep_cma_es.Sep_CMA_ES

::: evosax.algorithms.distribution_based.simple_es.SimpleES

::: evosax.algorithms.distribution_based.simulated_annealing.SimulatedAnnealing

::: evosax.algorithms.distribution_based.snes.SNES

::: evosax.algorithms.distribution_based.sv.sv_cma_es.SV_CMA_ES

::: evosax.algorithms.distribution_based.sv.sv_open_es.SV_Open_ES

::: evosax.algorithms.distribution_based.xnes.xNES

## Population-based strategies

::: evosax.algorithms.population_based.differential_evolution.DifferentialEvolution

::: evosax.algorithms.population_based.diffusion_evolution.DiffusionEvolution

::: evosax.algorithms.population_based.gesmr_ga.GESMR_GA

::: evosax.algorithms.population_based.learned_ga.LearnedGA

::: evosax.algorithms.population_based.mr15_ga.MR15_GA

::: evosax.algorithms.population_based.pso.PSO

::: evosax.algorithms.population_based.samr_ga.SAMR_GA

::: evosax.algorithms.population_based.simple_ga.SimpleGA

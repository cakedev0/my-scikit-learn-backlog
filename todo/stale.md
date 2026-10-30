# Threads waiting on me for 3+ weeks

For each line: tick it once handled on the spot (answered, closed), move it
to `TODO.md` when it needs real work, or delete it to drop it. `/todo-sync`
adds threads here when they've waited on me for more than three weeks, and
removes the ticked ones.

## My PRs

- [ ] Linear: [sklearn#35030](https://github.com/scikit-learn/scikit-learn/pull/35030) has conflicts (center X in GLMs, fixes [sklearn#35029](https://github.com/scikit-learn/scikit-learn/issues/35029): silently wrong coefficients on off-centered data), since 2026-09-25
- [ ] Threads: [sklearn#35090](https://github.com/scikit-learn/scikit-learn/pull/35090) has a red CI (warning filters in thread-based `Parallel`, fixes [sklearn#35089](https://github.com/scikit-learn/scikit-learn/issues/35089)), since 2026-10-07
- [ ] Preprocessing: jeremiedbb on [sklearn#34858](https://github.com/scikit-learn/scikit-learn/pull/34858) (QuantileTransformer/RobustScaler densify sparse columns; see also [sklearn#34857](https://github.com/scikit-learn/scikit-learn/issues/34857), [sklearn#34298](https://github.com/scikit-learn/scikit-learn/issues/34298)), since 2026-10-05
- [ ] Linear: [benchmarks#31](https://github.com/probabl-ai/scikit-learn-benchmarks/pull/31) has conflicts (Ridge centering trick benchmark)
- [ ] Trees: AnneBeyer on [sklearn#34267](https://github.com/scikit-learn/scikit-learn/pull/34267) (WIP deprecate the `ExtraTree*` classes), since 2026-06-13
- [ ] Upstream: seberg on [numpy#30066](https://github.com/numpy/numpy/pull/30066) (weighted nanquantile broadcasting), since 2026-01-06
- [ ] Array API: [fork#3](https://github.com/cakedev0/scikit-learn/pull/3) (`xpx.quantile` instead of `_weighted_percentile`), untouched since 2025-10
- [ ] Trees: [fork#4](https://github.com/cakedev0/scikit-learn/pull/4) (GB line search regression), untouched since 2026-01

## My reports with replies

- [ ] Upstream: david-cortes-intel on [oneDAL#3820](https://github.com/uxlfoundation/oneDAL/issues/3820) (cross_entropy_loss changes between calls with 4+ threads), since 2026-09-30
- [ ] Benchmarks: Vika-F and david-cortes-intel on [benchmarks#129](https://github.com/probabl-ai/scikit-learn-benchmarks/issues/129) (sklearnex TBB threads not limited under joblib), since 2026-10-08
- [ ] Benchmarks: david-cortes-intel mentioned me on [benchmarks#93](https://github.com/probabl-ai/scikit-learn-benchmarks/issues/93) (Anaconda in the envs matrix), since 2026-10-09
- [ ] Threads: itamarst on [joblib#1869](https://github.com/joblib/joblib/issues/1869) (loky workers' thread count vs physical cores), since 2026-10-01
- [ ] Maintenance: StefanieSenger on [sklearn#35047](https://github.com/scikit-learn/scikit-learn/issues/35047) (BaggingClassifier with feature sub-sampling and routed `X_val`), since 2026-10-02
- [ ] Encoders: lorentzenchr and thomasjpfan on [sklearn#34387](https://github.com/scikit-learn/scikit-learn/issues/34387) (OrdinalEncoder treats missing values as unknown categories first), since 2026-09-25
- [ ] Benchmarks: lucascolley on [sklearn#34865](https://github.com/scikit-learn/scikit-learn/issues/34865) (Meson downgrades -O3 to -O2 under conda-forge CFLAGS), since 2026-09-04
- [ ] GPU: virchan on [probabl-ai/scikit-learn-intel-workflow#8](https://github.com/probabl-ai/scikit-learn-intel-workflow/issues/8) (keep the float32 GPU runner?), since 2026-08-22
- [ ] Trees: adam2392 on [sklearn#34691](https://github.com/scikit-learn/scikit-learn/issues/34691) (RFC unknown categories vs missing values in HGB and trees), since 2026-08-14
- [ ] Trees: ogrisel on [sklearn#33158](https://github.com/scikit-learn/scikit-learn/issues/33158) (RFC why trees accept a `splitter` parameter), since 2026-06-12
- [ ] Trees: [sklearn#32752](https://github.com/scikit-learn/scikit-learn/issues/32752) (doc of tie-breaking between equivalent splits; contributors asking to work on it), since 2026-06-07
- [ ] Trees: adrinjalali, betatim and jjerphan on [sklearn#32384](https://github.com/scikit-learn/scikit-learn/issues/32384) (RFC is optimizing trees worth the complexity?), since 2025-12-15

## Mentions on other threads

- [ ] Trees: adrinjalali on [sklearn#34558](https://github.com/scikit-learn/scikit-learn/pull/34558) (harden tree loading against corrupted files), since 2026-09-18
- [ ] Encoders: itamarst on [sklearn#34383](https://github.com/scikit-learn/scikit-learn/issues/34383) (faster string representation than `object` in encoders), since 2026-09-28
- [ ] Encoders: [sklearn#32161](https://github.com/scikit-learn/scikit-learn/issues/32161) (OrdinalEncoder sorted by decreasing frequency), since 2026-08-18
- [ ] Encoders: ranykamel on [sklearn#34445](https://github.com/scikit-learn/scikit-learn/pull/34445) and [sklearn#11379](https://github.com/scikit-learn/scikit-learn/issues/11379) (mixed-type columns in encoders), since 2026-08-12


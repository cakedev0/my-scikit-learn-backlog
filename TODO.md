# Todo

My work todo list as a scikit-learn maintainer at Probabl, mostly on the
[Intel / Probabl collaboration](https://blog.probabl.ai/intel-and-probabl-announce-collaboration-to-accelerate-scikit-learn-on-intel-hardware)
to make scikit-learn faster on Intel hardware. How it's kept:
[todo/INSTRUCTIONS.md](todo/INSTRUCTIONS.md).

## Inbox

## Now

- [ ] Blog: tell jjerphan on [blog#228](https://github.com/scikit-learn/blog/pull/228) which envs use the pthreads and the OpenMP OpenBLAS builds, explicitly on the dashboard, and answer david-cortes-intel's new suggestions (mkl-service for MKL threads, wording of the free-threading speedup) (lorentzenchr's review addressed, benchmarks progress post)
- [ ] Maintenance: follow up on [sklearn#35254](https://github.com/scikit-learn/scikit-learn/issues/35254): settle option B (classes weighted by n_k / n) with dherrera1911 and betatim, so omarboukherys can open the PR (LDA solvers disagree with custom priors; lll-hhh and omarboukherys ask to confirm the direction)
- [ ] Maintenance: re-review [sklearn#35261](https://github.com/scikit-learn/scikit-learn/pull/35261) (1.10 deprecations in MDS; FrancoisPgm addressed my comments, approved by betatim)
- [ ] Array API: review [sklearn#35053](https://github.com/scikit-learn/scikit-learn/pull/35053) (store all fitted attributes as NumPy arrays; betatim asks for reviews for 1.10 and fixed some of what I reported, virchan asks how to fix the `CalibratedClassifierCV` and `cross_val_predict` regressions; since 2026-10-07)
- [ ] Linear: open an issue for sparse coordinate descent diverging on constant columns, as lorentzenchr asks on [sklearn#34339](https://github.com/scikit-learn/scikit-learn/pull/34339) (fix in [fork#40](https://github.com/cakedev0/scikit-learn/pull/40); he will xfail the tests in his PR)
- [ ] HGB: resolve the conflicts on [sklearn#34935](https://github.com/scikit-learn/scikit-learn/pull/34935) and find a second reviewer, jjerphan has no time (mitigation heuristics for HGB scalability; approved by ogrisel), dashboards in [benchmarks#81](https://github.com/probabl-ai/scikit-learn-benchmarks/pull/81)

## Reviews

- [ ] Maintenance: review [sklearn#28681](https://github.com/scikit-learn/scikit-learn/pull/28681) (refactor `_convert_container`; self-requested 2026-10-28, approved by adrinjalali)
- [ ] Maintenance: review [sklearn#26423](https://github.com/scikit-learn/scikit-learn/pull/26423) (stratified k-fold for multilabel data; self-requested 2026-10-28)
- [ ] HGB: review [sklearn#26311](https://github.com/scikit-learn/scikit-learn/pull/26311) (post-fit calibration for HGBT; self-requested 2026-10-28)
- [ ] Docs: review [sklearn#34852](https://github.com/scikit-learn/scikit-learn/pull/34852) (example of a custom GroupTimeSeriesSplit; self-requested 2026-10-28)
- [ ] Threads: review [sklearn#35062](https://github.com/scikit-learn/scikit-learn/pull/35062) (don't assume the CPU count never changes; itamarst updated it; since 2026-10-05)
- [ ] HGB: review [sklearn#34398](https://github.com/scikit-learn/scikit-learn/pull/34398) (Huber loss for HGB; mentioned by lorentzenchr)
- [ ] Maintenance: re-review [sklearn#35133](https://github.com/scikit-learn/scikit-learn/pull/35133) (explicit zeros in sparse DictVectorizer output; LJimmy007 says my requested changes are addressed)
- [ ] Maintenance: re-review [sklearn#34619](https://github.com/scikit-learn/scikit-learn/pull/34619) (faster `PCA(n_components="mle")`; jberg5 addressed my comments)
- [ ] Preprocessing: take over or review [sklearn#32806](https://github.com/scikit-learn/scikit-learn/pull/32806) (polars support in `make_column_selector`, maybe through narwhals as DeaMariaLeon suggested; AsherWright silent since my ping of 2026-10-05)

## Waiting

- [ ] Trees: get [sklearn#35181](https://github.com/scikit-learn/scikit-learn/pull/35181) reviewed (shared X validation and categorical encoding for trees, forests and GB; answered adam2392 and antoinebaker)
- [ ] HGB: [sklearn#35182](https://github.com/scikit-learn/scikit-learn/pull/35182) (split early stopping data after binning; draft, ogrisel commented)
- [ ] KMeans: get [sklearn#35193](https://github.com/scikit-learn/scikit-learn/pull/35193) reviewed (Lloyd thread buffers)
- [ ] Linear: [sklearn#35260](https://github.com/scikit-learn/scikit-learn/pull/35260) (sparse incremental mean and variance in float64)
- [ ] Linear: [benchmarks#156](https://github.com/probabl-ai/scikit-learn-benchmarks/pull/156) (fit time vs suboptimality, scikit-learn vs glum)
- [ ] Benchmarks: [benchmarks#141](https://github.com/probabl-ai/scikit-learn-benchmarks/pull/141) (Pareto fronts of XGBoost, LightGBM and CatBoost vs HGB; ogrisel commented)
- [ ] Threads: findings reported to scikit-learn, waiting on decisions: [sklearn#35091](https://github.com/scikit-learn/scikit-learn/issues/35091) (lock contention with free-threading), [sklearn#34764](https://github.com/scikit-learn/scikit-learn/issues/34764) (OpenMP active wait), [sklearn#34437](https://github.com/scikit-learn/scikit-learn/issues/34437) (PyPI wheel bundles the wrong libgomp)
- [ ] Maintenance: follow up on [sklearn#34994](https://github.com/scikit-learn/scikit-learn/pull/34994) (random projections `get_feature_names_out`; approved, waiting for the author to mark it ready)
- [ ] KMeans: get [sklearn#35311](https://github.com/scikit-learn/scikit-learn/pull/35311) merged (tolerance without `np.var` on dense data, from [fork#39](https://github.com/cakedev0/scikit-learn/pull/39); approved by ogrisel)
- [ ] Maintenance: get [sklearn#35295](https://github.com/scikit-learn/scikit-learn/pull/35295) merged (make `_safe_indexing` public as `safe_indexing`, fixes [sklearn#27439](https://github.com/scikit-learn/scikit-learn/issues/27439); conflicts resolved and `doc/conf.py` change dropped, approved by lorentzenchr)
- [ ] Maintenance: follow up on [sklearn#35302](https://github.com/scikit-learn/scikit-learn/pull/35302) (ignore the agents' instruction files in `.gitignore`; I asked to add `CLAUDE.md` and `AGENTS.override.md`, lorentzenchr points to `AGENTS.md`, I answered why)

## Later

- [ ] HGB: make the tools developed for [sklearn#34935](https://github.com/scikit-learn/scikit-learn/pull/34935) available in threadpoolctl
- [ ] HGB: faster single-threaded HGB, what LightGBM does differently: [fork#33](https://github.com/cakedev0/scikit-learn/issues/33), [fork#34](https://github.com/cakedev0/scikit-learn/pull/34) (skip unused bins in non-root histograms)
- [ ] HGB: categorical features, upstream or close [fork#30](https://github.com/cakedev0/scikit-learn/pull/30) (trees' categorical encoding in HGB) and [fork#32](https://github.com/cakedev0/scikit-learn/pull/32) (categorical and missing values in GradientBoosting), after [sklearn#35181](https://github.com/scikit-learn/scikit-learn/pull/35181)
- [ ] KMeans: upstream or close [fork#41](https://github.com/cakedev0/scikit-learn/pull/41) (float32 predict far from the origin) → [notes](issues/cluster-kmeans-predict-float32-not-centered.md)
- [ ] KMeans: maybe a blog post on the KMeans story (k-means++ and Lloyd scaling)
- [ ] Docs: Intel section in the performance and parallelism guides (which build to install, how to set threads), from laptops to data center servers
- [ ] Trees: binning in `RandomForest*` and `ExtraTrees*`, the main reason sklearnex forests are faster; prototype in [fork#6](https://github.com/cakedev0/scikit-learn/pull/6)
- [ ] Trees: faster sorting for exact splits and free-threaded forest fit, upstream or close [sklearn#34693](https://github.com/scikit-learn/scikit-learn/pull/34693) (SIMD sort), [fork#17](https://github.com/cakedev0/scikit-learn/pull/17) (3-way partitioning), [fork#19](https://github.com/cakedev0/scikit-learn/pull/19) (thread pool for forest fit), benchmarks [benchmarks#121](https://github.com/probabl-ai/scikit-learn-benchmarks/pull/121), [benchmarks#126](https://github.com/probabl-ai/scikit-learn-benchmarks/pull/126), [benchmarks#137](https://github.com/probabl-ai/scikit-learn-benchmarks/pull/137), [benchmarks#140](https://github.com/probabl-ai/scikit-learn-benchmarks/pull/140)
- [ ] Encoders: go back to the encoders work (paused)
  - [ ] answer lorentzenchr and thomasjpfan on [sklearn#34387](https://github.com/scikit-learn/scikit-learn/issues/34387) (OrdinalEncoder treats missing values as unknown categories first; since 2026-09-25)
  - [ ] answer itamarst's mention on [sklearn#34383](https://github.com/scikit-learn/scikit-learn/issues/34383) (faster string representation than `object` in encoders)
  - [ ] answer the mention on [sklearn#32161](https://github.com/scikit-learn/scikit-learn/issues/32161) (OrdinalEncoder sorted by decreasing frequency)
  - [ ] answer ranykamel on [sklearn#34445](https://github.com/scikit-learn/scikit-learn/pull/34445) and [sklearn#11379](https://github.com/scikit-learn/scikit-learn/issues/11379) (mixed-type columns in encoders)
  - [ ] resume or close the pandas categorical fast path [sklearn#34678](https://github.com/scikit-learn/scikit-learn/pull/34678) (every edge case as on main), with [sklearn#34721](https://github.com/scikit-learn/scikit-learn/pull/34721) (narwhals) and [narwhals#3761](https://github.com/narwhals-dev/narwhals/issues/3761) (access categorical codes)
  - [ ] upstream or close [fork#13](https://github.com/cakedev0/scikit-learn/pull/13) and [fork#15](https://github.com/cakedev0/scikit-learn/pull/15) (fast transform of small batches), benchmarks in [benchmarks#76](https://github.com/probabl-ai/scikit-learn-benchmarks/pull/76)
- [ ] Linear: find the copies of X that can be avoided (slow, single-threaded, double the peak memory)
- [ ] Linear: upstream or close [fork#21](https://github.com/cakedev0/scikit-learn/pull/21) (F-ordered X for multiclass lbfgs), [fork#37](https://github.com/cakedev0/scikit-learn/pull/37) (newton-cd in `solver="auto"`), [fork#38](https://github.com/cakedev0/scikit-learn/pull/38) (lbfgs on implicitly centered columns, replaces the closed [sklearn#35030](https://github.com/scikit-learn/scikit-learn/pull/35030) for [sklearn#35029](https://github.com/scikit-learn/scikit-learn/issues/35029), where lorentzenchr recalls that `normalize` was removed in favor of a `StandardScaler` pipeline), [fork#40](https://github.com/cakedev0/scikit-learn/pull/40) (sparse CD diverges on constant columns, see the issue to open for [sklearn#34339](https://github.com/scikit-learn/scikit-learn/pull/34339))
- [ ] Preprocessing: upstream or close [fork#22](https://github.com/cakedev0/scikit-learn/pull/22) and [fork#25](https://github.com/cakedev0/scikit-learn/pull/25) (faster `SplineTransformer.transform`)
- [ ] Threads: [fork#28](https://github.com/cakedev0/scikit-learn/pull/28) (multi-threaded input validation and preprocessing, WIP)
- [ ] Upstream: upstream to SciPy or close [cakedev0/scipy#4](https://github.com/cakedev0/scipy/pull/4) (faster weighted `stats.quantile`) and [cakedev0/scipy#5](https://github.com/cakedev0/scipy/pull/5) (faster `_xp_searchsorted`)
- [ ] Benchmarks: add `PCA`, `LogisticRegressionCV` and `RidgeCV`
- [ ] Benchmarks: discrete Intel GPUs (Arc B580), and more CUDA results to compare them with
- [ ] Benchmarks: a new Phoronix scikit-learn workload from a subset of the cases, with the Phoronix maintainers
- [ ] Benchmarks: run-to-run variability on multi-socket runners [benchmarks#80](https://github.com/probabl-ai/scikit-learn-benchmarks/issues/80), OpenMP active wait polluting later timings [benchmarks#90](https://github.com/probabl-ai/scikit-learn-benchmarks/issues/90), intel-laptop throttling [benchmarks#145](https://github.com/probabl-ai/scikit-learn-benchmarks/issues/145)
- [ ] Benchmarks: sparse datasets from pre-processed text [benchmarks#4](https://github.com/probabl-ai/scikit-learn-benchmarks/issues/4)
- [ ] GPU: move the Intel workflows to scikit-learn's CI? [probabl-ai/scikit-learn-intel-workflow#7](https://github.com/probabl-ai/scikit-learn-intel-workflow/issues/7)
- [ ] Trees: follow up on my RFCs: [sklearn#34691](https://github.com/scikit-learn/scikit-learn/issues/34691) (unknown categories vs missing values in HGB and trees; adam2392 replied), [sklearn#33158](https://github.com/scikit-learn/scikit-learn/issues/33158) (why trees accept `splitter`; ogrisel replied), [sklearn#32384](https://github.com/scikit-learn/scikit-learn/issues/32384) (is optimizing trees worth the complexity?; adrinjalali, betatim, jjerphan replied)
- [ ] Trees: follow up on [sklearn#32752](https://github.com/scikit-learn/scikit-learn/issues/32752) (doc of tie-breaking between equivalent splits): let one of the contributors asking take it, or close
- [ ] Trees: revive or close [sklearn#34267](https://github.com/scikit-learn/scikit-learn/pull/34267) (deprecate the `ExtraTree*` classes; WIP, answer AnneBeyer)
- [ ] Trees: upstream or close [fork#4](https://github.com/cakedev0/scikit-learn/pull/4) (GB line search regression)
- [ ] Array API: upstream or close [fork#3](https://github.com/cakedev0/scikit-learn/pull/3) (`xpx.quantile` instead of `_weighted_percentile`)
- [ ] Upstream: revive or close [numpy#30066](https://github.com/numpy/numpy/pull/30066) (weighted nanquantile broadcasting; answer seberg)
- [ ] Upstream: upstream to OpenBLAS or close [cakedev0/OpenBLAS#3](https://github.com/cakedev0/OpenBLAS/pull/3) (fewer threads in threaded syrk instead of one), [cakedev0/OpenBLAS#8](https://github.com/cakedev0/OpenBLAS/pull/8) (fewer threads for small dot, axpy, symv, getrf and gesv; draft) and [cakedev0/OpenBLAS#10](https://github.com/cakedev0/OpenBLAS/pull/10) (split k between threads in gemm and syrk when C is small)
- [ ] Upstream: report [cakedev0/cython#1](https://github.com/cakedev0/cython/issues/1) to Cython (`prange` in a nogil `parallel()` block lost its end barrier in 3.3, code after the loop races) with its fix [cakedev0/cython#3](https://github.com/cakedev0/cython/pull/3), and [cakedev0/cython#2](https://github.com/cakedev0/cython/issues/2) (`prange` holding the GIL hangs with clang); upstream or close [cakedev0/cython#4](https://github.com/cakedev0/cython/pull/4) (no barrier at the start of `prange` when it can't be observed); check scikit-learn's `prange` blocks
- [ ] Upstream: work on or report to OpenBLAS the LAPACK gaps with MKL: [cakedev0/OpenBLAS#5](https://github.com/cakedev0/OpenBLAS/issues/5) (tall `dgeqrf`, 7-15x), [cakedev0/OpenBLAS#6](https://github.com/cakedev0/OpenBLAS/issues/6) (`dpotrf`, ~3x), [cakedev0/OpenBLAS#7](https://github.com/cakedev0/OpenBLAS/issues/7) (`dsyevd`, ~2.3x); and [cakedev0/OpenBLAS#9](https://github.com/cakedev0/OpenBLAS/issues/9) (pinned threads sometimes share a CPU on the Xeon socket, 10-40x slower calls)
- [ ] HGB: upstream or close [fork#42](https://github.com/cakedev0/scikit-learn/pull/42) after [sklearn#34935](https://github.com/scikit-learn/scikit-learn/pull/34935) (merge the parallel regions of a split)

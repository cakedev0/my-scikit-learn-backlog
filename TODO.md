# Todo

My work todo list as a scikit-learn maintainer at Probabl, mostly on the
[Intel / Probabl collaboration](https://blog.probabl.ai/intel-and-probabl-announce-collaboration-to-accelerate-scikit-learn-on-intel-hardware)
to make scikit-learn faster on Intel hardware. How it's kept:
[todo/INSTRUCTIONS.md](todo/INSTRUCTIONS.md).

## Inbox

## Now

- [ ] Maintenance: follow up on [sklearn#32728](https://github.com/scikit-learn/scikit-learn/pull/32728): open the PR or the issue to make `_safe_indexing` public, without the underscore, as I offered (ogrisel agrees, lorentzenchr reopened the PR until then)
- [ ] Docs: answer ArturoAmorQ's doubt about the "stability across algorithms" wording on [sklearn#35267](https://github.com/scikit-learn/scikit-learn/pull/35267) (Consensus Index TODO in the clustering guide)
- [ ] Trees: answer the questions of adam2392 and antoinebaker on [sklearn#35181](https://github.com/scikit-learn/scikit-learn/pull/35181) (shared X validation and categorical encoding for trees, forests and GB)
- [ ] Linear: answer lorentzenchr's inline concerns on [sklearn#35196](https://github.com/scikit-learn/scikit-learn/pull/35196): the Hessian squares the condition number, and is always computing the eigenvalues cheap? (minimum norm newton-cholesky, alternative to [sklearn#34339](https://github.com/scikit-learn/scikit-learn/pull/34339))
- [ ] Threads: answer lesteve on [sklearn#35105](https://github.com/scikit-learn/scikit-learn/issues/35105) (RFC deprecate `pre_dispatch`)
- [ ] Benchmarks: answer Vika-F and david-cortes-intel on [benchmarks#129](https://github.com/probabl-ai/scikit-learn-benchmarks/issues/129) (sklearnex TBB threads not limited under joblib; since 2026-10-08)
- [ ] Benchmarks: tell david-cortes-intel whether the OMP runtime issue still happens with the latest conda-forge packages, on [benchmarks#93](https://github.com/probabl-ai/scikit-learn-benchmarks/issues/93) (Anaconda in the envs matrix; since 2026-10-09)
- [x] GPU: answer virchan on [probabl-ai/scikit-learn-intel-workflow#8](https://github.com/probabl-ai/scikit-learn-intel-workflow/issues/8) (keep the float32 GPU runner?; since 2026-08-22)

## Reviews

- [ ] Maintenance: review [sklearn#28681](https://github.com/scikit-learn/scikit-learn/pull/28681) (refactor `_convert_container`; self-requested 2026-10-28, approved by adrinjalali)
- [ ] Maintenance: review [sklearn#26423](https://github.com/scikit-learn/scikit-learn/pull/26423) (stratified k-fold for multilabel data; self-requested 2026-10-28)
- [ ] HGB: review [sklearn#26311](https://github.com/scikit-learn/scikit-learn/pull/26311) (post-fit calibration for HGBT; self-requested 2026-10-28)
- [ ] Docs: review [sklearn#34852](https://github.com/scikit-learn/scikit-learn/pull/34852) (example of a custom GroupTimeSeriesSplit; self-requested 2026-10-28)
- [ ] Threads: review [joblib#1899](https://github.com/joblib/joblib/pull/1899) (faster threading backend for `Parallel`, review requested by itamarst, who has since extended the tests)
- [ ] Threads: review [sklearn#35062](https://github.com/scikit-learn/scikit-learn/pull/35062) (don't assume the CPU count never changes; itamarst updated it)
- [ ] HGB: review [sklearn#34398](https://github.com/scikit-learn/scikit-learn/pull/34398) (Huber loss for HGB; mentioned by lorentzenchr)
- [ ] Linear: review [sklearn#35215](https://github.com/scikit-learn/scikit-learn/pull/35215) (user guide on linear models, intro and OLS; lorentzenchr updated it)
- [ ] Array API: review [sklearn#35053](https://github.com/scikit-learn/scikit-learn/pull/35053) (store all fitted attributes as NumPy arrays; betatim, ogrisel and david-cortes-intel commented)
- [ ] Maintenance: review [sklearn#35259](https://github.com/scikit-learn/scikit-learn/pull/35259) (skip Python 3.15 for Pyodide wheels; mentioned)
- [ ] Maintenance: re-review [sklearn#35133](https://github.com/scikit-learn/scikit-learn/pull/35133) (explicit zeros in sparse DictVectorizer output; LJimmy007 says my requested changes are addressed)
- [ ] Linear: follow up on [sklearn#32991](https://github.com/scikit-learn/scikit-learn/pull/32991): close it (lorentzenchr suggests it) or review betatim's refresh [sklearn#35247](https://github.com/scikit-learn/scikit-learn/pull/35247) (global_random_seed in the coordinate descent tests)
- [ ] Trees: answer adrinjalali's mention on [sklearn#34558](https://github.com/scikit-learn/scikit-learn/pull/34558) (harden tree loading against corrupted files; since 2026-09-18)

## Waiting

- [ ] HGB: get [sklearn#35216](https://github.com/scikit-learn/scikit-learn/pull/35216) merged (binning crash on all-missing features, fixes [sklearn#35164](https://github.com/scikit-learn/scikit-learn/issues/35164); approved by FrancoisPgm, needs a second review)
- [ ] HGB: get [sklearn#34935](https://github.com/scikit-learn/scikit-learn/pull/34935) merged (mitigation heuristics for HGB scalability; approved by ogrisel), dashboards in [benchmarks#81](https://github.com/probabl-ai/scikit-learn-benchmarks/pull/81)
- [ ] HGB: [sklearn#35182](https://github.com/scikit-learn/scikit-learn/pull/35182) (split early stopping data after binning; draft, ogrisel commented)
- [ ] KMeans: get [sklearn#35193](https://github.com/scikit-learn/scikit-learn/pull/35193) reviewed (Lloyd thread buffers), benchmarks in [benchmarks#154](https://github.com/probabl-ai/scikit-learn-benchmarks/pull/154) and [benchmarks#160](https://github.com/probabl-ai/scikit-learn-benchmarks/pull/160); with [OpenBLAS#6105](https://github.com/OpenMathLib/OpenBLAS/pull/6105) it should close [sklearn#17334](https://github.com/scikit-learn/scikit-learn/issues/17334)
- [ ] Linear: [sklearn#34793](https://github.com/scikit-learn/scikit-learn/pull/34793) (Ridge centering trick for cholesky; approved by lorentzenchr), benchmarks in [benchmarks#115](https://github.com/probabl-ai/scikit-learn-benchmarks/pull/115) and [benchmarks#31](https://github.com/probabl-ai/scikit-learn-benchmarks/pull/31) (has conflicts)
- [ ] Linear: [sklearn#35260](https://github.com/scikit-learn/scikit-learn/pull/35260) (sparse incremental mean and variance in float64)
- [ ] Linear: [benchmarks#156](https://github.com/probabl-ai/scikit-learn-benchmarks/pull/156) (fit time vs suboptimality, scikit-learn vs glum)
- [ ] Benchmarks: [benchmarks#141](https://github.com/probabl-ai/scikit-learn-benchmarks/pull/141) (Pareto fronts of XGBoost, LightGBM and CatBoost vs HGB; ogrisel commented)
- [ ] Blog: [blog#228](https://github.com/scikit-learn/blog/pull/228) (benchmarks progress post; approved by DeaMariaLeon) and [blog#231](https://github.com/scikit-learn/blog/pull/231) (match Substack's font size)
- [ ] Threads: findings reported to scikit-learn, waiting on decisions: [sklearn#35091](https://github.com/scikit-learn/scikit-learn/issues/35091) (lock contention with free-threading), [sklearn#34764](https://github.com/scikit-learn/scikit-learn/issues/34764) (OpenMP active wait), [sklearn#34736](https://github.com/scikit-learn/scikit-learn/issues/34736) (oversubscription in `pairwise_distances`), [sklearn#34437](https://github.com/scikit-learn/scikit-learn/issues/34437) (PyPI wheel bundles the wrong libgomp)
- [ ] Maintenance: follow up on [sklearn#34673](https://github.com/scikit-learn/scikit-learn/issues/34673) (regression, zero `sample_weight` in IsolationForest and Bagging, attracting automated PRs; asked antoinebaker to fix it)
- [ ] Maintenance: follow up on [sklearn#34994](https://github.com/scikit-learn/scikit-learn/pull/34994) (random projections `get_feature_names_out`; approved, waiting for the author to mark it ready)
- [ ] Upstream: sklearnex reports [sklearnex#3402](https://github.com/uxlfoundation/scikit-learn-intelex/issues/3402) (`estimators_` rebuilds every tree), [sklearnex#3377](https://github.com/uxlfoundation/scikit-learn-intelex/issues/3377) (CPU Ridge up to 55x slower when n_features > n_samples), [sklearnex#3220](https://github.com/uxlfoundation/scikit-learn-intelex/issues/3220) (LogisticRegression GPU fallback)
- [ ] Upstream: [data-apis/array-api#976](https://github.com/data-apis/array-api/issues/976) (why `stable=True` is the default for `xp.sort`)

## Later

- [ ] HGB: make the tools developed for [sklearn#34935](https://github.com/scikit-learn/scikit-learn/pull/34935) available in threadpoolctl
- [ ] HGB: faster single-threaded HGB, what LightGBM does differently: [fork#33](https://github.com/cakedev0/scikit-learn/issues/33), [fork#34](https://github.com/cakedev0/scikit-learn/pull/34) (skip unused bins in non-root histograms)
- [ ] HGB: categorical features, upstream or close [fork#30](https://github.com/cakedev0/scikit-learn/pull/30) (trees' categorical encoding in HGB) and [fork#32](https://github.com/cakedev0/scikit-learn/pull/32) (categorical and missing values in GradientBoosting), after [sklearn#35181](https://github.com/scikit-learn/scikit-learn/pull/35181)
- [ ] KMeans: upstream or close [fork#39](https://github.com/cakedev0/scikit-learn/pull/39) (tolerance without `np.var`) and [fork#41](https://github.com/cakedev0/scikit-learn/pull/41) (float32 predict far from the origin) → [notes](issues/cluster-kmeans-predict-float32-not-centered.md)
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
- [ ] Linear: upstream or close [fork#21](https://github.com/cakedev0/scikit-learn/pull/21) (F-ordered X for multiclass lbfgs), [fork#37](https://github.com/cakedev0/scikit-learn/pull/37) (newton-cd in `solver="auto"`), [fork#38](https://github.com/cakedev0/scikit-learn/pull/38) (lbfgs on implicitly centered columns, replaces the closed [sklearn#35030](https://github.com/scikit-learn/scikit-learn/pull/35030) for [sklearn#35029](https://github.com/scikit-learn/scikit-learn/issues/35029)), [fork#40](https://github.com/cakedev0/scikit-learn/pull/40) (sparse CD diverges on constant columns)
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

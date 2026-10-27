# ClassifierChain with `chain_method="decision_function"` or `"predict_log_proba"` and `cv=None` trains and predicts on features of different scales

```json
{
    "estimators": ["ClassifierChain"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["output-change", "unclear-spec"]
}
```

## Classification

- **Priority (2)**: silent and large accuracy loss (Jaccard 0.45 -> 0.13 on a simple dataset) for a documented parameter combination: `cv=None` is the default, and `chain_method` lists `decision_function` and `predict_log_proba` as valid choices. It is not the default `chain_method`, which keeps the impact limited.
- **Difficulty (2)**: the code change is small, but someone has to pick what the "true label" feature should be for each `chain_method`.
- **Decision (1)**: changing what the chain is trained on changes results for those settings (`output-change`). There is no single obvious mapping from labels to decision values, so the spec must be decided first (`unclear-spec`): either convert the true labels into the scale of `chain_method`, or raise/warn when `cv=None` is combined with a non-probability `chain_method`, or document it.

## Upstream search

- `gh issue list --search "ClassifierChain chain_method"`: no issue.
- `gh pr list --search "ClassifierChain chain_method"`: only #27700 (merged, adds `chain_method`). I read its review discussion: it covers `predict_log_proba` support and response-method priorities, but not the mismatch between training features (true labels) and prediction features when `cv=None`.
- `gh issue list --search "ClassifierChain random_state"` and other chain queries: #13339, #29856, #21761, #26817 are about other symptoms (error messages, NaN support, `predict_proba` columns, scorers).

## Describe the bug and its user-facing impact

With `cv=None` (the default), `ClassifierChain.fit` appends the **true labels** (0/1) of the previous outputs as extra features. At predict time, it appends the output of `chain_method` of the previous estimators instead. For `chain_method="predict"` both are 0/1, and for `"predict_proba"` both live in [0, 1]. But for `"decision_function"` the predict-time features are unbounded margins (e.g. -5 or +8), and for `"predict_log_proba"` they are in (-inf, 0], where a positive label gets ~0 and a negative label a large negative value. The downstream estimators were fitted on a completely different feature distribution, so their predictions are garbage.

The user sees no error or warning, only much worse predictions. With `cv=5` the training features come from `cross_val_predict(..., method=chain_method)`, so the scales match and the results are fine, which shows the problem is the `cv=None` path.

## Steps/Code to Reproduce

```python
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain

X, Y = make_multilabel_classification(n_samples=2000, n_classes=6, random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

for chain_method in ["predict", "predict_proba", "predict_log_proba", "decision_function"]:
    for cv in [None, 5]:
        chain = ClassifierChain(
            LogisticRegression(max_iter=1000), chain_method=chain_method, cv=cv
        ).fit(X_train, Y_train)
        score = jaccard_score(Y_test, chain.predict(X_test), average="samples", zero_division=0)
        print(f"{chain_method:>17}, cv={cv}: {score:.3f}")
```

## Expected Results

All settings should give a Jaccard score in the same ballpark (~0.45-0.50), as `chain_method` is described as only choosing which prediction is fed to the next estimators.

## Actual Results

```
          predict, cv=None: 0.446
          predict, cv=5: 0.476
    predict_proba, cv=None: 0.473
    predict_proba, cv=5: 0.479
predict_log_proba, cv=None: 0.147
predict_log_proba, cv=5: 0.503
decision_function, cv=None: 0.134
decision_function, cv=5: 0.477
```

## Root cause and possible fix

`sklearn/multioutput.py:721` (`_BaseChain.fit`): when `self.cv is None`, `X_aug` is built from `Y[:, self.order_]`, whatever `chain_method` is. `_get_predictions` (`sklearn/multioutput.py:663`) then feeds `chain_method` outputs at predict time.

Possible fixes:

- raise (or warn) when `cv=None` and `chain_method` is `"decision_function"` or `"predict_log_proba"`, asking for `cv`;
- or map the true labels to the scale of `chain_method` (e.g. 0/1 -> log(eps)/0 for `predict_log_proba`), which has no natural answer for `decision_function`;
- or at least document that `cv=None` only makes sense with `"predict"`/`"predict_proba"`.

The first option looks simplest to me, but it is a behavior change for code that runs today.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproduced on local checkout f570497251, no diff in the relevant code)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```

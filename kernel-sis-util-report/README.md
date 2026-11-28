# Draft kernel report: SIS_UTIL limit on hybrid x86

Draft of a regression report for the Linux kernel scheduler list. Not sent.

- `report.txt`: the email (plain text, 76 columns). The block above the
  `-----` line is a status note for reviewers, not part of the email.
- `spin_pair.c`: the reproducer attached to the email.
- `desktop-results/`: raw output of the runs on the Core Ultra 9 285
  desktop (mainline v7.1, v7.2 and v7.4-rc3). For v7.3 and Ubuntu 7.0.0-31
  the raw files were lost and only a summary is left.

Context: this comes from the investigation of slow all-core OpenMP runs in
scikit-learn on hybrid Intel CPUs
([scikit-learn#34935](https://github.com/scikit-learn/scikit-learn/pull/34935),
[loky#629](https://github.com/joblib/loky/issues/629)).

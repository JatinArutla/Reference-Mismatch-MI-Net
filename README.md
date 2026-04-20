# Refshift paper repo skeleton v1

This is the first clean paper-only skeleton for the EEG reference-mismatch benchmark.

What is frozen here:
- paper scope = benchmark-first
- datasets = IV-2a, OpenBMI, Cho2017, Dreyer2023
- native per-dataset montages
- target sampling rate = 250 Hz
- provisional bandpass = 8-32 Hz
- reference operators = native, car, median, gs, laplacian(knn4), bipolar(nearest)
- versioned channel/position artifacts and graph artifacts

What is intentionally NOT implemented yet:
- raw dataset loaders
- cache builder
- ATCNet trainer
- CSP+LDA runner
- benchmark runners

Why:
the goal is to lock the data contracts and preprocessing/reference contracts before
adding training code.

## Next implementation order
1. build preprocessed-subject cache
2. implement dataset loaders against `specs/datasets/*.yaml`
3. implement one benchmark runner that produces a full 6x6 matrix in one process
4. add smoke tests
5. run IV-2a end to end first


## Final frozen graph-position policy

The repository uses a single explicit source hierarchy for electrode geometry:

- **IV-2a**: MOABB `DigMontage`
- **OpenBMI**: MOABB `DigMontage`
- **Cho2017**: MOABB `DigMontage`
- **Dreyer2023**: `standard_1005` fallback by channel name

Reason:
- IV-2a, OpenBMI, and Cho2017 expose usable nonzero EEG coordinates through the MOABB-loaded
  `Raw` object, while auxiliary channels remain unpositioned.
- Dreyer2023 and Dreyer2023A expose channel names and types through MOABB, but no usable montage
  (`raw.get_montage() is None` and zero nonzero positions for all channels). So Dreyer keeps the
  standardized fallback geometry.

This policy is frozen now to avoid rerunning the benchmark later because of graph drift.

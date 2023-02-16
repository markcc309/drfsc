# Changelog

Version 0.0.2 (2023-02-16)

general:

- moved formatting to `src layout` in line with [PyPA User Guide](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)

drfsc.py:

- Added `DRFSC.set_initial_mu()` method to set initial mu value for the RFSC optimisation. See [DRFSC API Docs](drfsc_api.md) for details. `RFSC.upweight` removed as a consequence.

utils.py:

- renamed `vertical_distribution` to `feature_distribution`.
- renamed `horizontal_distribution` to `sample_distribution`.
- fixed indexing bug in `balanced_sample_partition()` which was causing not all samples to be included in the generated partitions.

# drfsc

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- [![PyPI version](add here)(url)] -->

An open-source library for a distributed randomised feature selection and classification algorithm.

## Code

[drfsc](https://github.com/markcc309/drfsc)

## Contributors

[Mark Chiu Chong](https://github.com/markcc309)

## Overview

`drfsc` is an open-source Python implementation of the Distributed Randomised Feature Selection algorithm for Classification problems (D-RFSC). Beside addressing some of the shortcomings of the conventional FS method, its good performance has previously been shown on a range of benchmark datasets. However, to date no Python implementation is available. `drfsc` offers an easy to use, parallelized probabilistic population-based feature selection scheme that is flexible and can be adapted to a wide range of binary classification problems and is particularly useful for large data problems where model interpretability and model explainability is of high importance. It provides modules for model fitting, evaluation, and visualization. Tutorial notebooks are provided to demonstrate the use of the package.

## Installation

The easiest way to install is from PyPI: just use

`pip install drfsc`

To install from source: clone this git repo, enter the directory, and run

`python setup.py install`

## License

We invite anyone interested to use and modify this code under a MIT license.

## Dependencies

`drfsc` depends on the following packages:

- [numpy](https://numpy.org/)
- [scipy](https://www.scipy.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [statsmodels](https://www.statsmodels.org/stable/index.html)

## References

Brankovic, A., Falsone, A., Prandini, M., Piroddi, L. (2018). [A feature selection and classification algorithm based on randomized extraction of model populations](https://doi.org/10.1109/tcyb.2017.2682418)

Brankovic, A., Piroddi, L. (2019). [A distributed feature selection scheme with partial information sharing](https://doi.org/10.1007/s10994-019-05809-y)

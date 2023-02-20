# drfsc

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
version: 0.0.5

An open-source library for a distributed randomised feature selection and classification algorithm.

## Authors and Contributors:

[Mark Chiu Chong](https://github.com/markcc309), [Aida Brankovic](https://github.com/aibrank).

## Overview

`drfsc` is an open-source Python implementation of the Distributed Randomised Feature Selection algorithm for Classification problems (D-RFSC). Beside addressing some of the shortcomings of the conventional FS method, its good performance has previously been shown on a range of benchmark datasets. However, to date no Python implementation is available. `drfsc` offers an easy to use, parallelized probabilistic population-based feature selection scheme that is flexible and can be adapted to a wide range of binary classification problems and is particularly useful for large data problems where model interpretability and model explainability is of high importance. It provides modules for model fitting, evaluation, and visualization. Tutorial notebooks are provided to demonstrate the use of the package.

## Installation

The easiest way to install is from PyPI: just use

`pip install drfsc`

## License

We invite anyone interested to use and modify this code under a MIT license.

## Dependencies

`drfsc` depends on the following packages:

- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [statsmodels](https://www.statsmodels.org/stable/index.html)

## References

The package has been developed based on research that came out of the Polytechnical University of Milan. The interested reader is referred to [2] for details related to the distribution procedure, and to [1] for a more thorough mathematical overview and for experimental comparisons to various alternate feature selection methods.

[1] Brankovic, A., Falsone, A., Prandini, M., Piroddi, L. (2018). [A feature selection and classification algorithm based on randomized extraction of model populations](https://doi.org/10.1109/tcyb.2017.2682418)

[2] Brankovic, A., Piroddi, L. (2019). [A distributed feature selection scheme with partial information sharing](https://doi.org/10.1007/s10994-019-05809-y)

## Citations

This package is developed in CSIRO’s Australian e-Health Research Centre. If you use `drfsc` package in your research we would appreciate a citation to the appropriate paper(s):

 - For general use of `drfsc` package you can read/cite the original article.
 - For information/use of the Randomised Feature Selection and classification concept you can read/cite original article [1].
 - For information/use of the Distributed Feature Selection architecture with partial information you can read/cite original article [2].

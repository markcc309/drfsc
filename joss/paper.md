---
title: 'drfsc: A Python package for Distributed Randomized Feature Selection and Classification'
tags:
  - Python
  - Feature selection
  - model interpretability
  - model selection
  - classification
authors:
  - name: Mark Chiu Chong
    corresponding: true # (This is how to denote the corresponding author)
    orcid: 0000-0002-6940-6661
    equal-contrib: false
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Aida Brankovic
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    orcid: 0000-0001-7978-575X
    affiliation: 1
  - name: Luigi Piroddi
    orcid: 0000-0002-6063-8060
    affiliation: 3
affiliations:
 - name: CSIRO Australian e-Health Research Centre
   index: 1
 - name: School of Mathematics and Physics, The University of Queensland, St Lucia, QLD 4072, Australia
   index: 2
 - name: Department of Electronics, Information and Bioengineering, Politecnico di Milano
   index: 3
date: 09 January 2023
bibliography: paper.bib

---

# Summary

Standard classification methods that do not rely on feature selection  are prone  to suffer from overfitting on large datasets, learning ephemeral patterns in the data. Feature selection is hence frequently employed as a pre-processing step to select a subset of relevant features for use in model construction [@10.5555/944919.944968]. By retaining only the most relevant information for a given prediction task and discarding irrelevant or redundant features, feature selection often leads to both improved performance and increased model interpretability, where the latter is especially crucial in contexts such as health informatics. This paper presents `drfsc`, an open-source Python implementation of the Distributed Randomised Feature Selection algorithm for Classification problems (D-RFSC). `drfsc` offers an easy to use, parallelized probabilistic population-based feature selection scheme that is flexible and can be adapted to a wide range of binary classification problems, and is particularly useful for large data problems where model interpretability and model explainability is of high importance. It also allows the integration of domain knowledge into the process of model development. Besides addressing some of the shortcomings of the conventional feature selection methods, the good performance of D-RFSC has previously been shown on a range of benchmark datasets (see [@brankovic2019distributed] and [@7890437]).


# Statement of need


Feature selection is a common pre-processing step in building classification models. Libraries such as `sklearn.feature_selection` implement different feature selection methods that can be broadly classified into three main categories [@CHANDRASHEKAR201416]: univariate (e.g., correlation filtering), wrapper methods (e.g., Sequential Feature Selection) and regularization-based methods (e.g., Lasso regression). The general drawbacks of these available methods include overfitting, high computational costs, and poor scalability to problems with many features in addition to some of them being either prone to get stuck in local minima or do not account for feature interactions or cannot be applied if the number of samples is less than the number of features. In addition, except for the univariate filter methods, none of these methods allows the integration of domain knowledge in the process of model development, which in some applications such as healthcare may be desired. Some of these issues have been addressed in the work of [@brankovic2019distributed] and [@7890437]. However, to date no Python implementation is available. To bridge this gap, we introduce `drfsc` which implements a distributed feature selection architecture in combination with a randomised population-based algorithm.

<!-- describe how it works -->
`drfsc` is a Python package that provides an implementation of the D-RFSC algorithm and extends the partitioning logic to both features and observations. It operates by performing separate independent feature selection procedures on smaller random subsets of features obtained by vertical partitioning. Each sub-process selects statistically accurate features, which are then evaluated for their importance based on a defined performance metric. Locally optimal features are iteratively selected to be shared with other sub-processes until all locally optimal models are equal, or another defined stopping criterion is met. It also takes the advantage of the randomised nature of the local search algorithm and data splitting to allow ensemble model generation. [Multiprocessing](https://docs.python.org/3/library/multiprocessing.html) is also used for performance optimization. The algorithm iterates by pooling the locally optimal features and sharing them with all subsets, and repeating the search. The interested reader is referred to [@brankovic2019distributed] for details related to the distribution procedure, and to [@7890437] for a more thorough mathematical overview and for experimental comparisons to various alternate feature selection methods. 


<!-- drfsc is open source: briefly explain its use -->
`drfsc` is designed to be intuitive and easy to use for those familiar with `scikit-learn`. We employ an object-oriented framework, and provide modules for model fitting, evaluation, and visualization. The user can define the model type (e.g. single model or ensemble), as well as the search architecture by defining the number of vertical (feature) and horizontal (observation) partitions, several different modes for how features are shared between partitions, the maximum number of iterations, whether features should be re-shuffled at each iteration, the desired performance metric to optimize, number of CPU cores to use, and parameters related to data pre-processing e.g. generation of polynomial and interaction features of the desired degree. The user can also define parameters related to the local RFSC optimizer algorithm by passing updated parameters to the RFSC setter method.

Additional information regarding the available methods and a list
of the allowed values for each parameter can be found in the package [documentation](https://markcc309.github.io/drfsc/)

# License and dependencies
`drfsc` is a cross-platform software package written for Python 3.9+. It is free to use, distributed under the MIT license and is available on [PyPI](https://pypi.org/) via `pip`. It depends on [numpy](https://numpy.org/) (Harris et al., 2020) for basic array manipulations, [statsmodels](https://www.statsmodels.org/stable/index.html) (Seabold and Perktold, 2010)
for the underlying logistic model, [scikit-learn](https://scikit-learn.org/) (Pedregosa et al., 2011) for facilitating pre-processing of data and evaluation of model performance, and [matplotlib](https://matplotlib.org/) (Hunter, 2007)
for output visualizations. It accepts data as either numpy arrays or [pandas](https://pandas.pydata.org/) DataFrames. The source code is [available](https://github.com/markcc309/drfsc) on GitHub at and open to all contributions and suggestions.

# Documentation & Tutorials

In the accompanying [documentation](https://markcc309.github.io/drfsc/), we provide worked examples of applications of `drfsc` to example datasets:

- [fitting of a `single` model](https://markcc309.github.io/drfsc/notebooks/01_fitting_single/);
- [fitting of an `ensemble` model](https://markcc309.github.io/drfsc/notebooks/02_fitting_ensemble/);


# Acknowledgements

This work was funded by the CSIRO Australian e-Health Research Centre.

# References

[^drfsc]: [https://github.com/markcc309/drfsc](https://github.com/markcc309/drfsc)
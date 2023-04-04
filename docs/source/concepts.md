# Concepts

Unlike its predecessor NeuroBayes, Cyclic Boosting is not a neural network. 
According to several aspects, the model family that can be considered closest
relatives to Cyclic Boosting are Generalized Additive Models:
* support of different scenarios and target ranges and by means of
[link functions](https://cyclic-boosting.readthedocs.io/en/latest/cyclic_boosting.html#module-cyclic_boosting.link)
* optimization of factors by coordinate descent, similar to
[backfitting](https://en.wikipedia.org/wiki/Backfitting_algorithm)
* smoothing of factor distributions for each feature, similar to smoothing
terms

That means Cyclic Boosting is not a deep learning approach, but a blend of
concepts from several shallow machine learning algorithms. It shines on
structured, heterogeneous data sets, especially ones with categorical features
of high cardinality.

Binning
-------

The whole mechanism of Cyclic Boosting is based on the pre-requisite of 
[binning](https://cyclic-boosting.readthedocs.io/en/stable/cyclic_boosting.binning.html)
of the different features, set via [flags](https://cyclic-boosting.readthedocs.io/en/latest/cyclic_boosting.html#module-cyclic_boosting.flags):
* Categorical features retain their original categories (learning specific
factors for each of the bins, supporting categorical features with high
cardinality).
* Continuous features are discretized to either having same bin widths
(equidistant binning) or containing approximately the same number of
observations in each bin (equistatistics binning with different bin widths).
* Missing values are collected in a dedicated bin and the corresponding factor
is either learned or set to an uninformative, neutral value.

A loose resemblance might be seen with histogram binning in
[LightGBM](https://github.com/Microsoft/LightGBM).

Training Procedure
------------------

... coordinate descent
... [presentation](https://github.com/FelixWick/cyclic-boosting-presentation/blob/main/CB_PyCon23.pdf)
... with boosting-like update of factors (if aggregate)

The combination of binning and coordinate descent corresponds to a local
optimization, enabling low-bias predictions of rare observations.

... [learn rate](https://cyclic-boosting.readthedocs.io/en/latest/cyclic_boosting.html#module-cyclic_boosting.learning_rate)
(to not depend on feature ordering)

... update prior predictions

Individual Explainability
-------------------------

...

Smoothing
---------

...

[smoothing](https://cyclic-boosting.readthedocs.io/en/stable/cyclic_boosting.smoothing.html)

Interaction Terms
-----------------

...

Local optimization of interaction terms can be used to cover even rarer
observations.

Sample Weights
--------------

...

... also enables [background subtraction mode](https://cyclic-boosting.readthedocs.io/en/latest/cyclic_boosting.html#module-cyclic_boosting.GBSregression)
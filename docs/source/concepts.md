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

The empirical risk minimization in the Cyclic Boosting training is implemented
by a cyclic coordinate descent optimization algorithm (kind of
forward-stagewise modeling, aka boosting) minimizing the cost function of the
quadratic loss in each feature bin. (A more detailed explanation can be found
in this
[presentation](https://github.com/FelixWick/cyclic-boosting-presentation/blob/main/CB_PyCon23.pdf).)

The combination of binning and coordinate descent corresponds to a local
optimization, enabling low-bias predictions of rare observations.

Some details:
* The update of factors can either be done gradually (updating the value from
the previous iteration for each feature bin) or from scratch for each feature
in each iteration (depending on a model hyperparameter `aggregate`).
* A [learning rate](https://cyclic-boosting.readthedocs.io/en/latest/cyclic_boosting.html#module-cyclic_boosting.learning_rate)
can be applied if dependence on feature sequence is unwanted.
* Regularization is performed by means of a Bayesian factor update according to
conjugate prior distributions (e.g., gamma prior with Poisson likelihood or
beta prior with binomial likelihood).
* It is possible to update given prior predictions in a Bayesian way.

Smoothing
---------

Instead of using the factors estimated according to the fitting scheme above
directly, smoothed factors are used for subsequent optimization. For this,
typically generic functions (e.g., orthogonal polynomials) are fitted to the
factor distributions of in the different bins of each feature. Separate
smoothings are applied to each feature in each iteration.

This somothing procedure can be considered a kind of regularizatio and helps to
avoid overfitting (drastically reducing variance) by ignoring statistical
fluctuations in the factors.

Choosing specific [smoothing](https://cyclic-boosting.readthedocs.io/en/stable/cyclic_boosting.smoothing.html)
functions (e.g., monotonic, sinusoidal, (piecewise) linear) can be used as a
way to include prior knowledge about the dependencies between different
features and the target values. 

Interaction Terms
-----------------

By means of multi-dimensional binning, it is straight-forward to construct
interaction terms, i.e., features composed of several original input variables
(e.g., two-dimensional or three-dimensional interactions). The factor
estimation is done in the same way as for one-dimensional features and
smoothing can, for example, be applied across one of the axes
([GroupBySmoother](https://cyclic-boosting.readthedocs.io/en/latest/cyclic_boosting.smoothing.html#cyclic_boosting.smoothing.multidim.GroupBySmoother)).

The local optimization in interaction terms can be used to cover even rarer
observations.

Sample Weights
--------------

Binning also allows a natural way to imply sample weights during the training
procedure. This can be very helpful when it is desirable to focus on specific
classes of observation, e.g., higher weighting of recent observations in time
series tasks.

By using negative sample weights, this also enables
[background subtraction mode](https://cyclic-boosting.readthedocs.io/en/latest/cyclic_boosting.html#module-cyclic_boosting.GBSregression).
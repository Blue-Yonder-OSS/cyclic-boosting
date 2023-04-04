# Introduction

Cyclic Boosting is a simple yet powerful machine learning method for structured
data. It is robust and fast, features individual explainability of predictions
(in the style of Generalized Additive Models) and good predictability of rare
observations, takes into account multiplicative effects in a natural way,
supports categorical features of high cardinality, and enables highly complex
models by means of interaction terms.

Furthermore, model building with Cyclic Boosting is very convenient:
* few hyperparameters to be tuned
* not much data pre-processing needed
* easily configurable for different data types
* supporting missing values in input data
* assisting model development with individual analysis plots for features

Instead of estimating parameters in the sense of coefficients in linear
regression or weights in neural networks, Cyclic Boosting estimates factors (or
summands in additive regression mode) for each bin of the different features.
This allows a natural interpretation of each individual prediction in terms of
the different feature values, i.e., individual explainability.

The Cyclic Boosting [base algorithm](https://cyclic-boosting.readthedocs.io/en/latest/cyclic_boosting.html#module-cyclic_boosting.base)
is quite generic and can be adapted for both regression (in several modes) and
classification tasks.

Regression
----------

The two main regression modes are [multiplicative regression](https://cyclic-boosting.readthedocs.io/en/latest/cyclic_boosting.html#module-cyclic_boosting.regression)
and [additive regression mode](https://cyclic-boosting.readthedocs.io/en/latest/cyclic_boosting.html#module-cyclic_boosting.location).
Topology-wise, the first resembles [Poisson regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html),
i.e., a Generalized Linear model with a log link function, covering the target
range $[0, \infty]$ and supporting multiplicative dependencies, while the
latter is similar to linear regression, covering the target range
$[-\infty, \infty]$ and supporting additive dependencies. It should be noted
though that, in contrast to Generalized Linear Models, Cyclic Boosting in
either mode enables highly non-linear models, similar to
[Generalized Additive Models](https://en.wikipedia.org/wiki/Generalized_additive_model).

Classification
--------------

The [classification mode](https://cyclic-boosting.readthedocs.io/en/latest/cyclic_boosting.html#module-cyclic_boosting.classification)
resembles logistic regression (Generalized Linear Model with logit link) and
predicts probabilities, i.e., covering the target range $[0, 1]$. It supports
multiplicative dependencies between features and target odds.

The Story behind Cyclic Boosting
--------------------------------

The idea of many of the building blocks of Cyclic Boosting, e.g., binning and
smoothing, originated from an algorithm called NeuroBayes, a feed-forward
neural network with lots of pre-processing steps, developed by Professor
Michael Feindt and some students from his experimental particle physics group
at University of Karlsruhe back in the 90s. (The Bayes part comes from the
interpretation of network outputs as a posteriori probabilities in
classificiation tasks, the usage of Bayesian statistics for regularization, and
taking into account a priori knowledge in the form of inclusive distributions.)

In a spin-off company called PhiI-T, which was later rebranded to Blue Yonder,
NeuroBayes was then used for various business problems, one of which was demand
forecasting for retailers. Although the NeuroBayes model was pretty successful
for this use case, it exhibited several shortcomings like lack of individual
explainability of predictions, difficulties to predict rare observations, or
dedicated treatment of mulitplicative effects. Therefore, Cyclic Boosting (to
be exact, its multiplicative regression mode) was developed as successor
algorithm to overcome these issues. Although initially, Cyclic Boosting was
developed specifically for retail demand forecasting, it is a general-purpose
algorithm that can also be employed for many other use cases, including
classification tasks.

Because demand forecasts are often used as input for order optimization
(replenishment), the prediction of full probability distributions, rather than
mere point estimates (typically the conditional mean for most machine learning
methods), is desirable to properly minimize realistic cost functions. For that
reason, a [width mode](https://cyclic-boosting.readthedocs.io/en/latest/cyclic_boosting.html#module-cyclic_boosting.nbinom)
was added to Cyclic Boosting to estimate individual negative binomial
distributions.

The price of products is one of the main influencing factors of demand, and
estimating individual price-demand elasticities can improve demand models
significantly. And the price-demand elasticities are also directly helpful for
another important use case in retail, namely dynamic pricing. For these
reasons, an [exponential elasticity mode](https://cyclic-boosting.readthedocs.io/en/latest/cyclic_boosting.html#module-cyclic_boosting.price)
was added to Cyclic Boosting.

Later, yet another [background subtraction mode](https://cyclic-boosting.readthedocs.io/en/latest/cyclic_boosting.html#module-cyclic_boosting.GBSregression),
was added for the use case of customer targeting, where a specific action like
coupon sending only affects a small portion of customers and the bulk of
unaffected customers needs to be statistically removed to identify causal
effects.

There is a bunch of people who contributed to the development of the Cyclic
Boosting library over the years. Without guarantee of completeness, here is a
list in alphabetic order: Bruno Daniel, Michael Feindt, Martin Hahn, Uwe Korn,
Holger Peters, Thomas Pfaff, Jörg Rittinger, Daniel Stemmer, Felix Wick
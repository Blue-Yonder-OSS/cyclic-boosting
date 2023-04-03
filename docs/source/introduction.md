# Introduction

Cyclic Boosting is a simple yet powerful machine learning method for structured
data. It is robust and fast, features individual explainability of predictions
(in the style of Generalized Additive Models) and good predictability of rare
observations, takes into account multiplicative effects in a natural way, and
enables highly complex models by means of interaction terms.

Also model building with Cyclic Boosting is very convenient:
* few hyperparameters to be tuned
* not much data pre-processing needed
* easily configurable for different data types
* supporting missing values in input data
* assisting model development with individual analysis plots for features

Regression
----------

...

Classification
--------------

...

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

... phit-t and Blue Yonder

... retail demand forecasting

... width mode for subsequent order optimization (replenishment)

... elasticity mode for dynamic pricing

... background subtraction mode for customer targeting

... contributors
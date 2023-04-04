# Concepts

Unlike its predecessor NeuroBayes, Cyclic Boosting is not a neural network. 
The model family that can be considered closest relatives to Cyclic Boosting
are Generalized Additive Models:
* ... topology ... [link functions](https://cyclic-boosting.readthedocs.io/en/latest/cyclic_boosting.html#module-cyclic_boosting.link)
* ... backfitting
* ... smoothing

That means Cyclic Boosting is not a deep learning approach, but a blend of
concepts from several shallow machine learning algorithms. It shines on
structured, heterogeneous data sets, especially ones with categorical features
of high cardinality.

Binning
-------

...

[binning](https://cyclic-boosting.readthedocs.io/en/stable/cyclic_boosting.binning.html)

... LGBM

Training Procedure
------------------

... coordinate descent with boosting-like update of factors

The combination of binning and coordinate descent corresponds to a local
optimization, enabling low-bias predictions of rare observations.

... aggregate

... [learn rate](https://cyclic-boosting.readthedocs.io/en/latest/cyclic_boosting.html#module-cyclic_boosting.learning_rate)

... update prior predictions

Individual Explainability
-------------------------

...

Smoothing
---------

...

[smoothing](https://cyclic-boosting.readthedocs.io/en/stable/cyclic_boosting.smoothing.html)

[flags](https://cyclic-boosting.readthedocs.io/en/latest/cyclic_boosting.html#module-cyclic_boosting.flags)

Interaction Terms
-----------------

...

Local optimization of interaction terms can be used to cover even rarer
observations.

Sample Weights
--------------

...

... also enables [background subtraction mode](https://cyclic-boosting.readthedocs.io/en/latest/cyclic_boosting.html#module-cyclic_boosting.GBSregression)

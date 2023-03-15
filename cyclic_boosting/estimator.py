from cyclic_boosting import CBPoissonRegressor, binning
from sklearn.pipeline import Pipeline

def CBPoissonRegressorEstimator(
        feature_groups=None,
        feature_properties=None,
        weight_column=None,
        prior_prediction_column=None,
        minimal_loss_change=1e-3,
        minimal_factor_change=1e-3,
        maximal_iterations=10,
        observers=None,
        smoother_choice=None,
        output_column=None,
        learn_rate=None,
        number_of_bins=100,
        aggregate=True,
    ):

    estimator=CBPoissonRegressor(
        feature_groups=feature_groups,
        feature_properties=feature_properties,
        weight_column=weight_column,
        prior_prediction_column=prior_prediction_column,
        minimal_loss_change=minimal_loss_change,
        minimal_factor_change=minimal_factor_change,
        maximal_iterations=maximal_iterations,
        observers=observers,
        smoother_choice=smoother_choice,
        output_column=output_column,
        learn_rate=learn_rate,
        aggregate=aggregate,
    )

    binner = binning.BinNumberTransformer(n_bins=number_of_bins, feature_properties=feature_properties)

    CBPoissonRegressorEstimator = Pipeline([("binning", binner), ("CB", estimator)])
    return CBPoissonRegressorEstimator
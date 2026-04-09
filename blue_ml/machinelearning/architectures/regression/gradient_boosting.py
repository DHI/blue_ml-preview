"""Gradient boosting regression models."""

from blue_ml.machinelearning.architectures.regression.base_class import (
    BlueRegressionModel,
)


class BlueGBoost(BlueRegressionModel):
    """Gradient boosting regression model with feature selection and hyperparameter tuning.

    This class provides a wrapper around scikit-learn's GradientBoostingRegressor
    with support for multi-output regression, recursive feature elimination (RFECV),
    and automated hyperparameter optimization using Optuna.

    Parameters
    ----------
    params : dict, optional
        Model configuration parameters. Expected keys:
        - gb_params : dict, optional
            Parameters to pass to sklearn.ensemble.GradientBoostingRegressor
        - multi_output : bool, optional
            Whether to wrap the estimator in MultiOutputRegressor for multiple targets
        - use_rfecv : bool, optional
            Whether to use recursive feature elimination with cross-validation
        - rfecv_params : dict, optional
            Parameters to pass to sklearn.feature_selection.RFECV
    """

    ...

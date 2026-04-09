"""Random forest regression models."""

from typing import Optional

from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]
from sklearn.feature_selection import RFECV  # type: ignore[import-untyped]

from blue_ml.machinelearning.architectures.regression.base_class import (
    BlueRegressionModel,
)


class BlueForest(BlueRegressionModel):
    """Random forest regression model with optional feature selection.

    This class provides a wrapper around scikit-learn's RandomForestRegressor
    with support for recursive feature elimination using cross-validation (RFECV)
    to automatically select the most important features.

    Parameters
    ----------
    params : dict, optional
        Model configuration parameters. Expected keys:
        - rf_params : dict, optional
            Parameters to pass to sklearn.ensemble.RandomForestRegressor
        - use_rfecv : bool, optional
            Whether to use recursive feature elimination with cross-validation
        - rfecv_params : dict, optional
            Parameters to pass to sklearn.feature_selection.RFECV
    """

    ...

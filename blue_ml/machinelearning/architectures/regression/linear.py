"""Linear regression models."""

from typing import Optional

import numpy as np
import optuna
from sklearn.feature_selection import RFECV  # type: ignore[import-untyped]
from sklearn.linear_model import LinearRegression, Ridge  # type: ignore[import-untyped]

from blue_ml.machinelearning.architectures.regression.base_class import (
    BlueRegressionModel,
)
from blue_ml.timeseries import Timeseries


class BlueLinearModel(BlueRegressionModel):
    """Linear regression model with optional recursive feature elimination.

    This class provides a wrapper around scikit-learn's LinearRegression model
    with support for recursive feature elimination using cross-validation (RFECV)
    to automatically select the most important features.

    Parameters
    ----------
    params : dict, optional
        Model configuration parameters. Expected keys:
        - lm_params : dict, optional
            Parameters to pass to sklearn.linear_model.LinearRegression
        - use_rfecv : bool, optional
            Whether to use recursive feature elimination with cross-validation
        - rfecv_params : dict, optional
            Parameters to pass to sklearn.feature_selection.RFECV
    """

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "This model is not implemented in the preview release."
        )


class BlueRidge(BlueRegressionModel):
    """Ridge regression model with optional feature selection and hyperparameter tuning.

    This class provides a wrapper around scikit-learn's Ridge regression model
    with support for recursive feature elimination (RFECV) and automated
    hyperparameter optimization using Optuna.

    Parameters
    ----------
    params : dict, optional
        Model configuration parameters. Expected keys:
        - ridge_params : dict, optional
            Parameters to pass to sklearn.linear_model.Ridge
        - use_rfecv : bool, optional
            Whether to use recursive feature elimination with cross-validation
        - rfecv_params : dict, optional
            Parameters to pass to sklearn.feature_selection.RFECV
    """

    ...

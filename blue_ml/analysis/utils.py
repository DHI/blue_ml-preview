"""Utility functions for statistical analysis and curve fitting."""

from typing import Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm  # type: ignore[import-untyped]
from sklearn.metrics import mean_squared_error  # type: ignore[import-untyped]
from statsmodels.tsa.stattools import ccf  # type: ignore[import-untyped]

from blue_ml.timeseries import Timeseries


def fit_smooth_curve(
    h: float,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    *,
    is_opt: bool = True,
    n: int = 100,
) -> Union[float, Tuple[np.ndarray, np.ndarray]]:
    """
    Fit a LOWESS smooth curve to data points.

    Parameters
    ----------
    h : float
        Bandwidth parameter for LOWESS smoothing (fraction of data used).
    x_vals : np.ndarray
        Independent variable values.
    y_vals : np.ndarray
        Dependent variable values.
    is_opt : bool, optional
        If True, return MSE for optimization. If False, return smoothed curve.
        Default is True.
    n : int, optional
        Number of points for the smoothed curve when is_opt=False.
        Default is 100.

    Returns
    -------
    float or tuple of np.ndarray
        If is_opt=True, returns mean squared error as float.
        If is_opt=False, returns tuple of (x_loess, y_loess) arrays.
    """
    if is_opt:
        qs = np.quantile(x_vals, q=(0.005, 0.995))
        idx_qs = np.where((x_vals > qs[0]) * (x_vals < qs[1]))
        y_loess = sm.nonparametric.lowess(
            y_vals[idx_qs], x_vals[idx_qs], frac=h, xvals=x_vals
        )
        return mean_squared_error(y_vals, y_loess)
    else:
        x_loess = np.linspace(x_vals.min(), x_vals.max(), n)
        y_loess = sm.nonparametric.lowess(y_vals, x_vals, frac=h, xvals=x_loess)
        return x_loess, y_loess


def crosscorrelation(
    x: Union[np.ndarray, pd.Series], y: Union[np.ndarray, pd.Series], nlags: int = 30
):
    """
    Compute cross-correlation between two time series.

    Parameters
    ----------
    x : np.ndarray or pd.Series
        First time series.
    y : np.ndarray or pd.Series
        Second time series.
    nlags : int, optional
        Number of lags to compute. Default is 30.

    Returns
    -------
    pd.Series
        Cross-correlation values indexed by lag.
    """
    if isinstance(x, pd.Series):
        x = np.asarray(x.values)

    if isinstance(y, pd.Series):
        y = np.asarray(y.values)

    ccf_left = ccf(y, x, nlags=nlags)[::-1][:-1]
    ccf_right = ccf(x, y, nlags=nlags)
    ccf_values = np.concatenate([ccf_left, ccf_right])
    lag_axis = np.arange(-(nlags - 1), nlags)
    return pd.Series(ccf_values, index=lag_axis)


def get_crosscorr_dataframe(
    x: Timeseries, cc_threshold: float = 0.4, max_n_lags: int = 48
):
    """
    Calculate cross-correlations between features and target in a Timeseries.

    Parameters
    ----------
    x : Timeseries
        Timeseries object containing features and targets.
    cc_threshold : float, optional
        Minimum absolute correlation value to include in results.
        Default is 0.4.
    max_n_lags : int, optional
        Maximum number of lags to compute. Default is 48.

    Returns
    -------
    pd.DataFrame
        DataFrame with cross-correlation values for features exceeding threshold.

    Raises
    ------
    AssertionError
        If targets are not 1-dimensional.
    """
    features = x.features
    targets = x.targets

    assert len(targets.names) == 1, "Only works with 1-D targets"
    y = targets.values.ravel()

    ccf_results = []
    for col in features.names:
        ccf_as_series = crosscorrelation(
            x=features.to_dataframe()[col], y=y, nlags=max_n_lags
        )
        if ccf_as_series.abs().max() > cc_threshold:
            ccf_as_series.name = col
            ccf_results.append(ccf_as_series)

    return pd.concat(ccf_results, axis=1)

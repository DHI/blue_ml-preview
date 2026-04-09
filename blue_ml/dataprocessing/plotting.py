"""Plotting utilities for data processing visualization."""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
from statsmodels.graphics.tsaplots import (  # type: ignore[import-untyped]
    plot_acf,
    plot_pacf,
)

from blue_ml.analysis.utils import crosscorrelation


def plot_acf_pacf(x, n_lags):
    """
    Plot autocorrelation and partial autocorrelation functions.

    Parameters
    ----------
    x : array-like
        Time series data.
    n_lags : int
        Number of lags to include in the plots.
    """
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 4))
    plot_acf(x, ax=axes[0], lags=n_lags)
    plot_pacf(x, ax=axes[1], lags=n_lags)

    plt.tight_layout()
    plt.show()


def plot_ccf(output, variables, n_lags=30, titles=None, ylim=(-0.9, 1)):
    """
    Plot cross-correlation functions between output and multiple variables.

    Parameters
    ----------
    output : array-like
        Output time series.
    variables : list
        List of variable time series to compare against output.
    n_lags : int, optional
        Number of lags to compute. Default is 30.
    titles : list of str or None, optional
        Titles for each subplot. If None, uses default titles.
        Default is None.
    ylim : tuple of float, optional
        Y-axis limits for plots. Default is (-0.9, 1).
    """
    n_subplots = len(variables)
    if not titles:
        titles = ["output"] + ["" for _ in range(n_subplots)]

    fig, axes = plt.subplots(
        1, n_subplots, figsize=((n_subplots * 3) + 1, 3), sharey=True
    )

    for i, var_title in enumerate(zip(variables, titles[1:])):
        var, title = var_title

        ccf_i = crosscorrelation(output, var, nlags=n_lags)

        idx_max_corr = np.argmax(np.abs(ccf_i))
        idx_in_series = ccf_i.index[idx_max_corr]
        value_in_series = ccf_i.values[idx_max_corr]
        segment_x = [ccf_i.index[idx_max_corr], ccf_i.index[idx_max_corr]]
        segment_y = [0, ccf_i.values[idx_max_corr]]

        axes[i].plot(ccf_i, zorder=1)
        axes[i].scatter(idx_in_series, value_in_series, zorder=2)
        axes[i].plot(segment_x, segment_y, c="darkgrey", zorder=0)

        axes[i].axvline(x=0, c="lightgrey", zorder=0)
        axes[i].axhline(y=0, c="lightgrey", zorder=0)
        axes[i].set_xlim(-n_lags, n_lags)
        axes[i].set_ylim(*ylim)
        axes[i].set_title(title)

    fig.suptitle(f"Crosscorrelation of {titles[0]} with:")

    plt.tight_layout()
    plt.show()


def plot_matched_results_with_plotly(data, modeldict, other_models=[]):
    """
    Plot model results comparison using Plotly.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing time series data with a time index.
    modeldict : dict
        Dictionary containing 'colname', 'name', and 'style' for the main model.
    other_models : list of dict, optional
        List of dictionaries for additional models to plot.
        Each dict should contain 'colname', 'name', and 'style'.
        Default is [].
    """
    time_index = data.index

    y = data[modeldict["colname"]]
    name = modeldict["name"]
    style = modeldict["style"]

    trace1 = go.Scatter(x=time_index, y=y, mode="lines", name=name, line=style)

    other_traces = []
    for modeldict in other_models:
        y = data[modeldict["colname"]]
        name = modeldict["name"]
        style = modeldict["style"]
        trace = go.Scatter(x=time_index, y=y, mode="lines", name=name, line=style)
        other_traces.append(trace)

    layout = go.Layout(
        title="Comparison", xaxis=dict(title="Index"), yaxis=dict(title="Values")
    )

    fig = go.Figure(data=[trace1] + other_traces, layout=layout)
    fig.show()

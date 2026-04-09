"""Plotting functions for data analysis and visualization."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize_scalar

from blue_ml.analysis import fit_smooth_curve


def plot_smooth_curve(
    results: pd.DataFrame,
    ref_item_label: str,
    print_h: bool = False,
    return_fig: bool = False,
):
    """
    Plot smooth curves comparing model residuals with LOWESS smoothing.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing reference item and model results to compare.
    ref_item_label : str
        Column name of the reference item for comparison.
    print_h : bool, optional
        If True, print optimized bandwidth parameter h. Default is False.
    return_fig : bool, optional
        If True, return the figure object. If False, display the plot.
        Default is False.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if return_fig=True, otherwise None.
    """

    def rescolname(m):
        return f"residuals_{m}"

    results = results.copy()

    models_to_compare = [col for col in results.columns if col != ref_item_label]
    rescolumns = []
    for m in models_to_compare:
        results[rescolname(m)] = np.array(results[ref_item_label]) - np.array(
            results[m]
        )
        rescolumns.append(rescolname(m))

    melted_results = results[[ref_item_label] + rescolumns].melt(
        id_vars=ref_item_label, value_name="residuals"
    )

    g = sns.jointplot(
        data=melted_results,
        x=ref_item_label,
        y="residuals",
        hue="variable",
        joint_kws=dict(alpha=0.6, marker="."),
    )

    for m in models_to_compare:
        res_label = rescolname(m)
        x_vals = np.asarray(results[ref_item_label].values)
        y_vals = np.asarray(results[res_label].values)

        # Create wrapper function for optimization that handles keyword arguments
        def objective_function(h: float) -> float:
            result = fit_smooth_curve(h, x_vals, y_vals, is_opt=True)
            # When is_opt=True, it should return a float
            assert isinstance(result, float), "Expected float return for optimization"
            return result

        h_opt = minimize_scalar(
            objective_function,
            bounds=(0.2, 0.9),  # type: ignore
        )
        if print_h:
            print(m, round(h_opt.x, 3))
        result = fit_smooth_curve(float(h_opt.x), x_vals, y_vals, is_opt=False, n=500)
        # When is_opt=False, it returns a tuple
        assert isinstance(result, tuple), "Expected tuple return when is_opt=False"
        x_loess, y_loess = result
        g.ax_joint.plot(x_loess, y_loess, lw=3)

    g.ax_joint.axhline(y=0, c="lightgrey", ls="dashed", zorder=-1)
    g.ax_joint.set_xlim(0, 5)

    g.ax_joint.grid(False)
    g.ax_marg_x.grid(False)
    g.ax_marg_y.grid(False)

    if return_fig:
        return g.figure
    else:
        plt.show()

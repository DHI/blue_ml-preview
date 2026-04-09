"""Wrapper for modelskill to assess model skill against observations."""

from typing import Any

import modelskill as ms
import pandas as pd
from pandas.io.formats.style import Styler

from blue_ml.timeseries.timeseries_item import TimeseriesItem

ms.load_style("MOOD")


def _parse_input(timeseries_item):
    if isinstance(timeseries_item, TimeseriesItem):
        return timeseries_item.to_dataframe()
    elif isinstance(timeseries_item, pd.DataFrame):
        return timeseries_item
    else:
        raise ValueError(
            f"Unsupported input type: {type(timeseries_item)}. "
            "Expected TimeseriesItem or pandas.DataFrame."
        )


def _format_skill(cc: Any) -> Styler:
    """
    Format and style skill metrics from a calibration/verification object.

    The input object ``cc`` is expected to support the method chain
    ``cc.skill().to_dataframe()``, which must return a :class:`pandas.DataFrame`
    containing at least the following numeric columns:

    - ``"n"``
    - ``"bias"``
    - ``"rmse"``
    - ``"urmse"``
    - ``"mae"``
    - ``"cc"``
    - ``"si"``
    - ``"r2"``

    The returned :class:`pandas.io.formats.style.Styler` applies numeric
    formatting, color gradients, and highlighting so the table can be rendered
    directly in notebooks.
    """
    df_skill = cc.skill().to_dataframe()

    abs_bias = df_skill["bias"].abs().max()

    df_skill_styled = (
        df_skill.style.format(
            {
                "n": "{:.0f}",
                "bias": "{:+.3f}",
                "rmse": "{:.3f}",
                "urmse": "{:.3f}",
                "mae": "{:.3f}",
                "cc": "{:.3f}",
                "si": "{:.3f}",
                "r2": "{:.3f}",
            }
        )
        .background_gradient(cmap="Greens", subset=["n"])
        .background_gradient(
            cmap="coolwarm", subset=["bias"], vmin=-abs_bias, vmax=abs_bias
        )
        .background_gradient(cmap="GnBu_r", subset=["rmse"])
        .background_gradient(cmap="GnBu_r", subset=["urmse"])
        .background_gradient(cmap="GnBu_r", subset=["mae"])
        .background_gradient(cmap="GnBu", subset=["cc"])
        .background_gradient(cmap="GnBu_r", subset=["si"])
        .background_gradient(cmap="GnBu", subset=["r2"])
        .highlight_max(subset=["cc", "r2"], props="font-weight: bold;")
        .highlight_min(
            subset=["rmse", "urmse", "mae", "si"], props="font-weight: bold;"
        )
    )

    return df_skill_styled


class ModelSkillAssessor:
    """Class to assess model skill against observations using modelskill."""

    def __init__(self, target=None, baseline=None, predictions=None):
        self.baseline = None
        self.target = None
        self.predictions = {}
        if target is not None:
            self.add_target(target)
        if baseline is not None:
            self.add_baseline(baseline)
        if predictions is not None:
            for name, prediction in predictions.items():
                self.add_prediction(prediction, name=name)

    def add_target(self, target, name="Target", quantity=None):
        self.target = ms.PointObservation(
            _parse_input(target), name=name, quantity=quantity
        )

    def add_baseline(self, baseline, name="Baseline", quantity=None):
        self.baseline = ms.PointModelResult(
            _parse_input(baseline), name=name, quantity=quantity
        )

    def add_prediction(self, prediction, name="Prediction", quantity=None):
        self.predictions[name] = ms.PointModelResult(
            _parse_input(prediction), name=name, quantity=quantity
        )

    @property
    def _modelskill_mod(self):
        lst = []
        if self.baseline:
            lst.append(self.baseline)
        if self.predictions:
            lst.extend(self.predictions.values())
        return lst

    def _match(self):
        self._cc = ms.match(obs=self.target, mod=self._modelskill_mod)

    def plot_timeseries(self):
        self._match()
        return self._cc.plot.timeseries(figsize=(10, 5))

    def plot_scatter(self):
        self._match()
        return self._cc.plot.scatter(skill_table=True, figsize=(5, 5))

    def skill_table(self, formatted=True):
        self._match()
        if formatted:
            return _format_skill(self._cc)
        return self._cc.skill().to_dataframe()


__all__ = ["ModelSkillAssessor"]

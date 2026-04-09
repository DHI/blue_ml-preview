"""Resampling operations for Timeseries.

This module handles temporal resampling of timeseries data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from blue_ml.timeseries.timeseries import Timeseries


class TimeseriesResampler:
    """Handles resampling operations for Timeseries.

    Responsibility: Resample timeseries data to different temporal frequencies.
    """

    @staticmethod
    def resample(
        ts: Timeseries,
        rule: str,
        method: Literal[
            "mean", "sum", "max", "min", "median", "std", "first", "last"
        ] = "mean",
    ) -> Timeseries:
        """Resample timeseries to a different frequency.

        Parameters
        ----------
        ts : Timeseries
            Timeseries to resample
        rule : str
            Resampling frequency rule (e.g., 'H' for hourly, 'D' for daily)
        method : str, default="mean"
            Aggregation method: 'mean', 'sum', 'max', 'min', 'median', 'std', 'first', 'last'

        Returns
        -------
        Timeseries
            Resampled timeseries

        Raises
        ------
        ValueError
            If method is not supported
        """
        from blue_ml.timeseries.timeseries import Timeseries

        supported_functions = [
            "mean",
            "sum",
            "max",
            "min",
            "median",
            "std",
            "first",
            "last",
        ]

        if method.lower() not in supported_functions:
            raise ValueError(
                f"Method '{method}' not supported. Supported methods: {supported_functions}"
            )

        df = ts.to_dataframe()

        # Using pandas resample since xarray's is too slow in the current implementation
        df = df.resample(rule).apply(method)

        ts_resampled = Timeseries.from_index_and_values(
            values=df.values,
            index=df.index,
            column_names=ts.names,
            target_names=ts.targets.names,
            attrs=ts.attrs,
        )

        return ts_resampled

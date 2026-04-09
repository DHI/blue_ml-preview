"""Utility functions for time series preprocessing."""

import xarray as xr
from darts import TimeSeries as DartsTimeSeries  # type: ignore[import-untyped]
from darts.utils.missing_values import (  # type: ignore[import-untyped]
    fill_missing_values,
)

from blue_ml.timeseries import Timeseries, TimeseriesData


def fill_gaps_with_darts(ts: Timeseries, freq: str) -> Timeseries:
    """Fill gaps in time series data using Darts library.

    Parameters
    ----------
    ts : Timeseries
        Time series data with potential gaps
    freq : str
        Frequency string for resampling (e.g., '1H', '1D')

    Returns
    -------
    Timeseries
        Time series with filled gaps
    """

    def fill_gaps_of_view(ts: TimeseriesData) -> xr.Dataset:
        datavars = ts.names
        attributes = {var: ts[var].attrs for var in datavars}

        dts = DartsTimeSeries.from_dataframe(
            ts.to_dataframe(), fill_missing_dates=True, freq=freq
        )
        dts = fill_missing_values(dts)

        ds = xr.Dataset.from_dataframe(dts.pd_dataframe())
        for var in datavars:
            ds[var].attrs = attributes[var]

        return ds

    filled_features = fill_gaps_of_view(ts.features)
    filled_targets = fill_gaps_of_view(ts.targets)

    return Timeseries(targets=filled_targets, features=filled_features)

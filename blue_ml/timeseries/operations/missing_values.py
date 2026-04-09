"""Missing value handling for Timeseries.

This module handles all missing value operations including filling, dropping, and gap detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Tuple, Union

import pandas as pd
import xarray as xr
from darts.utils.missing_values import fill_missing_values  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from blue_ml.timeseries.timeseries import Timeseries


class MissingValueHandler:
    """Handles missing value operations for Timeseries.

    Responsibility: Fill, drop, and detect missing values in timeseries data.
    """

    @staticmethod
    def fill_missing_values(
        ts: Timeseries,
        fill: Union[str, float] = "auto",
        **interpolate_kwargs,
    ) -> Timeseries:
        """Fill missing values in the timeseries.

        Wraps the `darts.utils.missing_values.fill_missing_values` method.

        Parameters
        ----------
        ts : Timeseries
            Timeseries to fill
        fill : str or float, default="auto"
            The value used to replace the missing values.
            If set to 'auto', will auto-fill missing values using pandas interpolation.
        **interpolate_kwargs
            Keyword arguments for pandas.DataFrame.interpolate()

        Returns
        -------
        Timeseries
            A new Timeseries with missing values filled

        Notes
        -----
        When targets are empty (prediction mode), only features are filled.
        """
        from blue_ml.timeseries.timeseries import Timeseries

        features_as_darts_ts = ts.features.to_darts()

        # Handle empty targets case
        if not ts.has_targets:
            filled_darts = fill_missing_values(
                features_as_darts_ts, fill=fill, **interpolate_kwargs
            )
            filled_features = filled_darts.to_dataframe()
            empty_targets = xr.Dataset(coords={"time": filled_features.index})
            return Timeseries(
                features=filled_features,
                targets=empty_targets,
                item_attrs=ts.attrs,
            )

        targets_as_darts_ts = ts.targets.to_darts()
        darts_ts = features_as_darts_ts.concatenate(
            targets_as_darts_ts, axis="component"
        )

        filled_darts = fill_missing_values(darts_ts, fill=fill, **interpolate_kwargs)

        return Timeseries.from_darts(
            filled_darts,
            feature_names=ts.features.names,
            target_names=ts.targets.names,
            attrs=ts.attrs,
        )

    @staticmethod
    def dropna(
        ts: Timeseries,
        how: Literal["any", "all"] = "any",
        thresh: Union[int, None] = None,
    ) -> Timeseries:
        """Drop missing values from timeseries.

        Parameters
        ----------
        ts : Timeseries
            Timeseries to process
        how : {'any', 'all'}, default='any'
            If 'any', drop the row if any of the values is missing.
            If 'all', drop the row if all of the values are missing.
        thresh : int, optional
            Require that many non-NA values

        Returns
        -------
        Timeseries
            A new Timeseries with missing values dropped

        Notes
        -----
        When targets are empty (prediction mode), only features are considered
        for determining which rows to drop.
        """
        from blue_ml.timeseries.timeseries import Timeseries

        # Only consider features for dropna when targets are empty
        if ts.has_targets:
            ds_to_check = ts.as_xarray()
        else:
            ds_to_check = ts.features.as_xarray()

        ds_without_na = ds_to_check.dropna(dim="time", how=how, thresh=thresh)

        # Get the valid time indices
        valid_times = ds_without_na.coords["time"]

        # Handle empty targets - create empty dataset with filtered time coord
        if not ts.has_targets:
            empty_targets = xr.Dataset(coords={"time": valid_times})
            return Timeseries(
                features=ds_without_na,
                targets=empty_targets,
                item_attrs=ts.attrs,
            )

        return Timeseries(
            data=ds_without_na,
            features=ts.features.names,
            targets=ts.targets.names,
            item_attrs=ts.attrs,
        )

    @staticmethod
    def detect_gaps(
        ts: Timeseries, mode: str = "all"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Detect gaps in features and targets.

        Parameters
        ----------
        ts : Timeseries
            Timeseries to analyze
        mode : str, default="all"
            Mode for gap detection

        Returns
        -------
        tuple of pd.DataFrame
            (features_gaps, targets_gaps)
        """
        features_gaps = ts.features.gaps(mode)
        targets_gaps = ts.targets.gaps(mode)
        return features_gaps, targets_gaps

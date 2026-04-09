"""Format conversion utilities for Timeseries data.

This module handles all format conversions between different data representations,
following the Single Responsibility Principle by separating conversion logic
from the core Timeseries data structure.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Union

import mikeio
import numpy as np
import pandas as pd
import xarray as xr
from darts.timeseries import TimeSeries as DartsTimeSeries  # type: ignore[import-untyped]

from blue_ml._utils import warn


class FormatConverter:
    """Handles conversion between different data formats.

    Responsibility: Convert data between numpy, pandas, xarray, mikeio, and darts formats.
    """

    _UNNAMED_TYPES = (np.ndarray, xr.DataArray, pd.Series)
    _UNINDEXED_TYPES = np.ndarray

    @staticmethod
    def _xa_to_ds(xa: xr.DataArray, name: Optional[str] = None) -> xr.Dataset:
        """Convert xarray DataArray to xarray Dataset.

        Parameters
        ----------
        xa : xr.DataArray
            Input data array with time dimension
        name : str, optional
            Variable name for the dataset, by default None

        Returns
        -------
        xr.Dataset
            Dataset containing the data array

        Raises
        ------
        ValueError
            If data is not 1D or missing time dimension
        """
        if len(xa.dims) > 1:
            raise ValueError("Only 1D data is supported")

        if "time" not in xa.dims:
            raise ValueError("Time dimension is missing")

        if name is not None:
            xa.name = name
        return xa.to_dataset()

    @staticmethod
    def _ds_to_ds(ds: xr.Dataset) -> xr.Dataset:
        """Validate and return xarray Dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Input dataset

        Returns
        -------
        xr.Dataset
            Validated dataset

        Raises
        ------
        ValueError
            If dataset is not 1D or missing time dimension
        """
        if len(ds.dims) > 1:
            raise ValueError("Only 1D data is supported")

        if "time" not in ds.dims:
            raise ValueError(f"'time' dimension is missing in {ds.dims}")
        return ds

    @staticmethod
    def _mikeiods_to_ds(mio_ds: mikeio.Dataset) -> xr.Dataset:
        """Convert mikeio Dataset to xarray Dataset.

        Parameters
        ----------
        mio_ds : mikeio.Dataset
            MIKE IO dataset

        Returns
        -------
        xr.Dataset
            Converted and validated dataset
        """
        ds = mio_ds.to_xarray()
        return FormatConverter._ds_to_ds(ds)

    @staticmethod
    def _mikeioda_to_ds(mio_da: mikeio.DataArray) -> xr.Dataset:
        """Convert mikeio DataArray to xarray Dataset.

        Parameters
        ----------
        mio_da : mikeio.DataArray
            MIKE IO data array

        Returns
        -------
        xr.Dataset
            Converted dataset
        """
        xa = mio_da.to_xarray()
        return FormatConverter._xa_to_ds(xa)

    @staticmethod
    def _df_to_ds(df: pd.DataFrame) -> xr.Dataset:
        """Convert pandas DataFrame to xarray Dataset.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with DatetimeIndex

        Returns
        -------
        xr.Dataset
            Converted dataset
        """
        df.index.name = "time"
        return df.to_xarray()

    @staticmethod
    def _nd_to_ds(
        data: np.ndarray,
        index: Optional[pd.Index],
        names: Optional[List[str]] = None,
    ) -> xr.Dataset:
        """Convert numpy array to xarray Dataset via DataFrame.

        Parameters
        ----------
        data : np.ndarray
            Array of values
        index : pd.Index, optional
            Time index for the data
        names : list of str, optional
            Variable names, by default None

        Returns
        -------
        xr.Dataset
            Converted dataset
        """
        df = pd.DataFrame(data, columns=names, index=index)
        return FormatConverter._df_to_ds(df)

    @staticmethod
    def _pdseries_to_ds(pd_series: pd.Series) -> xr.Dataset:
        """Convert pandas Series to xarray Dataset via DataFrame.

        Parameters
        ----------
        pd_series : pd.Series
            Series with DatetimeIndex

        Returns
        -------
        xr.Dataset
            Converted dataset
        """
        df = pd_series.to_frame()
        return FormatConverter._df_to_ds(df)

    @staticmethod
    def to_dataset(
        data: Union[
            np.ndarray,
            pd.DataFrame,
            pd.Series,
            xr.DataArray,
            xr.Dataset,
            mikeio.DataArray,
            mikeio.Dataset,
        ],
        names: Optional[List[str]] = None,
        time: Optional[pd.DatetimeIndex] = None,
    ) -> xr.Dataset:
        """Convert various data types to xarray Dataset.

        Accepts multiple data formats and converts them to a standardized
        xarray Dataset with time dimension.

        Parameters
        ----------
        data : np.ndarray, pd.DataFrame, pd.Series, xr.DataArray, xr.Dataset, mikeio.DataArray, or mikeio.Dataset
            Input data in any supported format
        names : list of str, optional
            Variable names (for unnamed types), by default None
        time : pd.DatetimeIndex, optional
            Time index (for unindexed types), by default None

        Returns
        -------
        xr.Dataset
            Standardized dataset with time dimension

        Raises
        ------
        ValueError
            If data type is not supported or required arguments are missing
        UserWarning
            If names or time are provided for already-named/indexed data
        """
        if (
            (not isinstance(data, FormatConverter._UNNAMED_TYPES))
            and (names is not None)
        ) or (
            isinstance(data, pd.Series)
            and data.name is not None
            and names is not None
            and names != [data.name]
        ):
            warn(
                f"The data passed is of type {type(data)}."
                "This type is already named, so the passed 'names' argument will be ignored.",
                UserWarning,
            )

        if (not isinstance(data, FormatConverter._UNINDEXED_TYPES)) and (
            time is not None
        ):
            warn(
                f"The data passed is of type {type(data)}."
                "This type already contains time index, so the passed 'time' argument will be ignored.",
                UserWarning,
            )

        if isinstance(data, np.ndarray):
            return FormatConverter._nd_to_ds(data, names=names, index=time)
        elif isinstance(data, pd.DataFrame):
            return FormatConverter._df_to_ds(data)
        elif isinstance(data, pd.Series):
            if names is None:
                if data.name is None:
                    raise ValueError(
                        "Series has no name, so 'names' arguments should not be empty"
                    )
            else:
                if len(names) == 1:
                    data.name = names[0]
                else:
                    raise ValueError("'names' argument should contain only one element")
            return FormatConverter._pdseries_to_ds(data)
        elif isinstance(data, xr.DataArray):
            name = None
            if names is not None:
                if len(names) == 1:
                    name = names[0]
                else:
                    raise ValueError("'names' argument should contain only one element")
            return FormatConverter._xa_to_ds(data, name)
        elif isinstance(data, xr.Dataset):
            return FormatConverter._ds_to_ds(data)
        elif isinstance(data, mikeio.Dataset):
            return FormatConverter._mikeiods_to_ds(data)
        elif isinstance(data, mikeio.DataArray):
            return FormatConverter._mikeioda_to_ds(data)
        elif hasattr(data, "_ds"):
            return FormatConverter._ds_to_ds(data._ds)  # type: ignore[attr-defined]
        else:
            raise ValueError(f"Cannot set item from data type {type(data)}")

    @staticmethod
    def get_data_names(
        data: Union[
            np.ndarray,
            pd.DataFrame,
            pd.Series,
            xr.DataArray,
            xr.Dataset,
            mikeio.DataArray,
            mikeio.Dataset,
        ],
    ) -> List[str]:
        """Extract variable names from data object.

        Parameters
        ----------
        data : np.ndarray, pd.DataFrame, pd.Series, xr.DataArray, xr.Dataset, mikeio.DataArray, or mikeio.Dataset
            Data object to extract names from

        Returns
        -------
        list of str
            Variable names, or ["unnamed"] if not available
        """
        if isinstance(data, mikeio.Dataset):
            return data.names
        elif isinstance(data, mikeio.DataArray):
            return [data.name or "unnamed"]
        elif isinstance(data, pd.DataFrame):
            return [str(col) for col in data.columns]
        elif isinstance(data, pd.Series):
            return [str(data.name) if data.name is not None else "unnamed"]
        elif isinstance(data, xr.Dataset):
            return [str(var) for var in data.data_vars]
        elif isinstance(data, xr.DataArray):
            return [str(data.name) if data.name is not None else "unnamed"]
        else:
            return ["unnamed"]

    @staticmethod
    def to_darts(ds: xr.Dataset) -> DartsTimeSeries:
        """Convert xarray Dataset to Darts TimeSeries.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to convert

        Returns
        -------
        DartsTimeSeries
            Converted Darts TimeSeries

        Raises
        ------
        UserWarning
            If time step is non-equidistant
        """

        def find_freq(time_coord: Union[xr.DataArray, np.ndarray]) -> str:
            # Convert to pandas DatetimeIndex for frequency inference
            if isinstance(time_coord, xr.DataArray):
                time_values: np.ndarray = time_coord.values
            else:
                time_values = time_coord

            time_index = pd.DatetimeIndex(time_values)
            inferred_freq = pd.infer_freq(time_index)

            if inferred_freq is None:
                n_obs = len(time_index)
                step_sizes_in_seconds = (
                    pd.Series(time_index)
                    .diff()
                    .dropna()
                    .apply(lambda x: x.total_seconds())
                )
                timestep_counts = step_sizes_in_seconds.value_counts()
                most_frequent_step_size = timestep_counts.idxmax()
                n_obs_1 = timestep_counts[most_frequent_step_size]
                timestep_counts.pop(most_frequent_step_size)

                inferred_freq = f"{int(most_frequent_step_size) // 3600}h"
                # We want to check whether the most frequent step size and its double take more
                # than 99% of the observations. The reason of including the double is that the interpolation
                # will be most reliable. In case they do not take the 99%, we will raise a warning.
                next_step_size = 2 * most_frequent_step_size
                if next_step_size in timestep_counts.index:
                    n_obs_2 = timestep_counts[next_step_size]
                else:
                    n_obs_2 = 0
                if (n_obs_1 + n_obs_2) / n_obs < 0.99:
                    fraction = 1 - round((n_obs_1 + n_obs_2) / n_obs, 1)
                    warnings.warn(
                        f"Time step is non-equidistant. {fraction}% of samples are separated by gaps "
                        f"wider than ({int(most_frequent_step_size)}) or ({int(next_step_size)})",
                        UserWarning,
                    )

            return inferred_freq

        xa_ = ds.to_dataarray("component")
        xa_ = xa_.expand_dims({"sample": 1}).transpose(
            "time", "component", "sample"
        )  # Correct order
        inferred_freq = find_freq(
            xa_["time"]
        )  # If no frequency is found, an error is raised by darts

        return DartsTimeSeries.from_xarray(xa_, freq=inferred_freq)

    @staticmethod
    def from_darts(darts_ts: DartsTimeSeries) -> xr.Dataset:
        """Convert Darts TimeSeries to xarray Dataset.

        Parameters
        ----------
        darts_ts : DartsTimeSeries
            Darts TimeSeries to convert

        Returns
        -------
        xr.Dataset
            Converted dataset
        """
        # Convert to xarray and reshape
        xa = darts_ts.to_xarray()
        # Remove sample dimension if it exists and has size 1
        if "sample" in xa.dims and xa.sizes["sample"] == 1:
            xa = xa.squeeze("sample")
        # Convert to dataset
        return xa.to_dataset(dim="component")

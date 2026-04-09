from __future__ import annotations

import os

os.environ["DARTS_CONFIGURE_MATPLOTLIB"] = "0"

from copy import deepcopy
from typing import TYPE_CHECKING, Any, List, Optional, Union, cast

import mikeio
import numpy as np
import pandas as pd
import xarray as xr


if TYPE_CHECKING:
    # These attributes are expected to exist on classes that use these mixins
    from typing import Protocol

    class TimeSeriesProtocol(Protocol):
        _ds: xr.Dataset
        names: List[str]
        values: np.ndarray
        time: pd.DatetimeIndex
        attrs: dict[str, Any]

        def as_xarray(self) -> Union[xr.Dataset, xr.DataArray]: ...
        def __getitem__(self, key: str) -> Any: ...


class _BaseFunctionsMixin:
    """Mixin providing basic operations for time series data.

    Implements arithmetic operations (+, -, *, /), equality comparison,
    copying, and empty state checking for time series classes.

    Attributes
    ----------
    _empty_ds : xr.Dataset
        Empty dataset template for comparison
    """

    _empty_ds = xr.Dataset(coords={"time": ("time", pd.to_datetime([]))})

    def copy(self):
        """Create a deep copy of the time series object.

        Returns
        -------
        Time series object
            Deep copy of self
        """
        return deepcopy(self)

    def __math_op__(self, other, op):
        # Parse other data
        other = self.__other_value__(other)
        # Arithmetic
        ts = self.copy()
        values = op(cast("TimeSeriesProtocol", self).values, other)
        cast("TimeSeriesProtocol", ts)._ds.values = values
        return ts

    def __other_value__(self, other):
        if isinstance(other, (int, float)):
            return np.full(cast("TimeSeriesProtocol", self).values.shape, other)
        elif isinstance(other, np.ndarray):
            return other
        elif isinstance(other, type(self)):
            # TODO: Improve. This workaround is used to be able to avoid
            # the circular import error
            return cast("TimeSeriesProtocol", other).values

        raise ValueError(f"Unsupported type {type(other)}")

    def __add__(self, other):
        return self.__math_op__(other, np.add)

    def __sub__(self, other):
        return self.__math_op__(other, np.subtract)

    def __mul__(self, other):
        return self.__math_op__(other, np.multiply)

    def __truediv__(self, other):
        return self.__math_op__(other, np.divide)

    def __eq__(self, other) -> bool:
        self_protocol = cast("TimeSeriesProtocol", self)
        other_protocol = cast("TimeSeriesProtocol", other)

        vars_are_equal = (
            len(
                set(self_protocol.names).symmetric_difference(set(other_protocol.names))
            )
            == 0
        )
        try:
            values_are_equal = all(
                [
                    np.allclose(self_protocol[name].values, other_protocol[name].values)
                    for name in self_protocol.names
                ]
            )
        except TypeError:
            # 'TimeseriesItem' is not subscriptable so we include this
            # to compare only values
            values_are_equal = np.allclose(self_protocol.values, other_protocol.values)

        time_is_equal = all(self_protocol.time == other_protocol.time)
        attrs_are_equal = self_protocol.attrs == other_protocol.attrs

        return vars_are_equal and values_are_equal and time_is_equal and attrs_are_equal

    @property
    def is_empty(self) -> bool:
        """Check if the time series contains no data.

        Returns
        -------
        bool
            True if the dataset is empty or has no variables
        """
        self_protocol = cast("TimeSeriesProtocol", self)
        # Use as_xarray() to get the dataset, as different classes store data differently
        ds = self_protocol.as_xarray()
        # Convert to Dataset if it's a DataArray for consistent comparison
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset()
        equals_default_empty_ds = ds.equals(self._empty_ds)
        has_no_variables = len(ds.data_vars) == 0
        return equals_default_empty_ds or has_no_variables


class _TSDataFormatMapperMixin:
    """Mixin for converting various data formats to xarray Dataset.

    DEPRECATED: This mixin is deprecated. Use FormatConverter directly instead.
    Kept for backward compatibility only.

    Provides methods to convert numpy arrays, pandas DataFrames/Series,
    xarray DataArray/Dataset, and mikeio DataArray/Dataset to a standard
    xarray Dataset format with time dimension.

    Attributes
    ----------
    _UNNAMED_TYPES : tuple
        Data types that require explicit naming
    _UNINDEXED_TYPES : type
        Data types that require explicit time index
    """

    _UNNAMED_TYPES = (np.ndarray, xr.DataArray, pd.Series)
    _UNINDEXED_TYPES = np.ndarray

    def _type_to_ds(
        self,
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
    ):
        """Convert various data types to xarray Dataset.

        DEPRECATED: Use FormatConverter.to_dataset() instead.
        """
        from blue_ml.timeseries.converters import FormatConverter

        return FormatConverter.to_dataset(data, names=names, time=time)

    @staticmethod
    def _get_data_names(
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

        DEPRECATED: Use FormatConverter.get_data_names() instead.
        """
        from blue_ml.timeseries.converters import FormatConverter

        return FormatConverter.get_data_names(data)


class _DataPropertiesMixin:
    """Mixin providing data property accessors for time series.

    Provides convenient properties for accessing time series dimensions,
    values, time index, frequency, and shape. Also implements conversion
    to pandas DataFrame.
    """

    @property
    def n_items(self) -> int:
        """Number of variables/items in the time series.

        Returns
        -------
        int
            Number of items
        """
        return len(cast("TimeSeriesProtocol", self).names)  # type: ignore[attr-defined]

    @property
    def n_time(self) -> int:
        """Number of time steps in the time series.

        Returns
        -------
        int
            Number of time steps
        """
        return len(cast("TimeSeriesProtocol", self).time)  # type: ignore[attr-defined]

    @property
    def values(self) -> np.ndarray:
        """Get values as numpy array.

        Returns
        -------
        np.ndarray
            Array of shape (n_time, n_items) containing all values
        """
        ds = cast("TimeSeriesProtocol", self).as_xarray()  # type: ignore[attr-defined]
        if isinstance(ds, xr.DataArray):
            return ds.T.values
        elif isinstance(ds, xr.Dataset):
            return ds.to_array().T.values
        else:
            # Fallback - should not happen if protocol is properly implemented
            raise ValueError("as_xarray() must return DataArray or Dataset")

    @property
    def time(self) -> pd.DatetimeIndex:
        """Get time index.

        Returns
        -------
        pd.DatetimeIndex
            Time index for the time series
        """
        ds = cast("TimeSeriesProtocol", self).as_xarray()  # type: ignore[attr-defined]
        return pd.DatetimeIndex(ds.time)

    @property
    def freq(self) -> Optional[str]:
        """Infer time series frequency.

        Returns
        -------
        str or None
            Inferred frequency string (e.g., 'H', 'D'), or None if irregular
        """
        time_index = cast("TimeSeriesProtocol", self).time  # type: ignore[attr-defined]
        return pd.infer_freq(time_index)

    @property
    def shape(self) -> tuple[int, int]:
        """Get shape of the time series.

        Returns
        -------
        tuple of int
            Shape as (n_time, n_items)

        Raises
        ------
        ValueError
            If data is not 2D
        """
        values = cast("TimeSeriesProtocol", self).values  # type: ignore[attr-defined]
        shape = values.shape
        if len(shape) != 2:
            raise ValueError(f"Expected 2D array, got {len(shape)}D")
        return (shape[0], shape[1])

    def __len__(self):
        return len(cast("TimeSeriesProtocol", self).values)  # type: ignore[attr-defined]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert time series to pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with DatetimeIndex and variable columns
        """
        ds = cast("TimeSeriesProtocol", self).as_xarray()  # type: ignore[attr-defined]
        return ds.to_dataframe()

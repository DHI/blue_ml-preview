"""Individual time series item representation."""

from __future__ import annotations

import os
from collections import OrderedDict
from enum import Enum
from typing import (
    Optional,
    Self,  # we dont support below 3.11, tough luck mypy
    Tuple,
    Union,
)

import mikeio
import numpy as np
import pandas as pd
import xarray as xr
from xarray.core.formatting_html import (
    _get_indexes_dict,
    _obj_repr,
    array_section,
    attr_section,
    coord_section,
    format_dims,
    index_section,
)

os.environ["DARTS_CONFIGURE_MATPLOTLIB"] = "0"

from blue_ml.config import CONFIG
from blue_ml.timeseries._mixins import (
    _BaseFunctionsMixin,
    _DataPropertiesMixin,
    _TSDataFormatMapperMixin,
)


class ItemAlias:
    """
    Class representing an alias for a timeseries item with variable and source.

    Parameters
    ----------
    variable : str
        Variable name.
    source : str
        Source identifier.
    """

    def __init__(self, variable: str, source: str):
        self.variable = variable
        self.source = source
        self.name = self._generate_name(variable, source)

    @classmethod
    def from_variable_name(cls, name: str) -> Self:
        """
        Create ItemAlias from a variable name string.

        Parameters
        ----------
        name : str
            Variable name, potentially containing source separator.

        Returns
        -------
        ItemAlias
            New ItemAlias instance.
        """
        if CONFIG.variable_source_separator in name:
            variable, source = tuple(name.split(CONFIG.variable_source_separator))
        else:
            variable = name
            source = ""
        return cls(variable, source)

    def _generate_name(self, variable: str, source: str) -> str:
        """Generate a variable name for the Timeseries object."""
        if CONFIG.variable_source_separator in source:
            return f"{variable}{CONFIG.variable_source_separator}{source}"
        else:
            return variable

    def update_name(
        self,
        suffix: Optional[str] = None,
        preffix: Optional[str] = None,
        appendix_sep: str = "_",
    ) -> str:
        """
        Update item name with prefix and/or suffix.

        Parameters
        ----------
        suffix : str or None, optional
            Suffix to append to variable name. Default is None.
        preffix : str or None, optional
            Prefix to prepend to variable name. Default is None.
        appendix_sep : str, optional
            Separator between name parts. Default is "_".

        Returns
        -------
        str
            Updated name.
        """
        variable, source = self.get_variable_and_source()

        if preffix:
            variable = f"{preffix}{appendix_sep}{variable}"
        if suffix:
            variable = f"{variable}{appendix_sep}{suffix}"

        return self._generate_name(variable, source)

    def get_variable_and_source(self) -> Tuple[str, str]:
        """
        Extract variable and source from the name.

        Returns
        -------
        tuple of str
            (variable, source) pair.

        Raises
        ------
        ValueError
            If name doesn't contain exactly 2 parts when using strict separator.
        """
        if CONFIG.variable_source_separator in self.name:
            parts = self.name.split(CONFIG.variable_source_separator)
            if len(parts) == 2:
                return (parts[0], parts[1])
            else:
                raise ValueError(
                    f"Expected exactly 2 parts when splitting '{self.name}' with separator '{CONFIG.variable_source_separator}'"
                )
        else:
            return self.name, ""

    def __str__(self):
        return self._generate_name(self.variable, self.source)

    def __repr__(self):
        return "'" + self.__str__() + "'"

    def __eq__(self, other):
        return self.__str__() == other

    def __hash__(self):
        return hash(self.__str__())


class ItemClass(Enum):
    """Enumeration of time series item classification types."""

    TARGET = "target"
    FEATURE = "feature"


class TimeseriesItem(
    _BaseFunctionsMixin, _TSDataFormatMapperMixin, _DataPropertiesMixin
):
    """
    Non-public xr.DataArray view class for item-based operations.

    Represents a single item (feature or target) in a timeseries.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series or xr.DataArray or mikeio.DataArray
        Input data for the item.
    name : str
        Name of the item.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, pd.Series, xr.DataArray, mikeio.DataArray],
        name: str,
    ):
        self._alias = ItemAlias.from_variable_name(name)

        ds = self._type_to_ds(data)
        self._ds = ds[name]
        self._attrs = ds.attrs[name] if name in ds.attrs else {}

    def _repr_html_(self) -> str:
        arr = self.as_xarray()
        dims = OrderedDict((k, v) for k, v in zip(arr.dims, arr.shape))
        if hasattr(arr, "xindexes"):
            indexed_dims = arr.xindexes.dims
        else:
            indexed_dims = {}

        obj_type = "<blue_ml.TimeseriesItem>"
        arr_name = f"'{arr.name}'" if getattr(arr, "name", None) else ""

        header_components = [
            f"<div class='xr-obj-type'>{obj_type}</div>",
            f"<div class='xr-array-name'>{arr_name}</div>",
            format_dims(dims, indexed_dims),
        ]

        sections = [array_section(arr)]

        if hasattr(arr, "coords"):
            sections.append(coord_section(arr.coords))

        if hasattr(arr, "xindexes"):
            indexes = _get_indexes_dict(arr.xindexes)
            sections.append(index_section(indexes))

        sections.append(attr_section(arr.attrs))

        return _obj_repr(arr, header_components, sections)

    def __repr__(self) -> str:
        return self.as_xarray().__repr__()

    @property
    def name(self) -> str:
        """Get the full name of the time series item.

        Returns
        -------
        str
            Full name combining variable and source
        """
        return self._alias.name

    @property
    def variable(self) -> str:
        """Get the variable name.

        Returns
        -------
        str
            Variable name
        """
        return self._alias.variable

    @property
    def source(self) -> str:
        """Get the source name.

        Returns
        -------
        str
            Source name
        """
        return self._alias.source

    @property
    def attrs(self) -> dict[str, dict]:
        """Get the attributes dictionary.

        Returns
        -------
        dict[str, dict]
            Attributes of the time series item
        """
        return self._ds.attrs

    def update_attrs(self, new_attrs: dict):
        """Update the attributes dictionary.

        Parameters
        ----------
        new_attrs : dict
            New attributes to add or update
        """
        self._ds.attrs.update(new_attrs)

    def to_series(self) -> pd.Series:
        """Convert to pandas Series.

        Returns
        -------
        pd.Series
            Time series as pandas Series
        """
        return self.as_xarray().to_series()

    def as_xarray(self) -> xr.DataArray:
        """Get the underlying xarray DataArray.

        Returns
        -------
        xr.DataArray
            The xarray DataArray representation
        """
        return self._ds

    def __eq__(self, other) -> bool:
        """Check equality with another TimeseriesItem.

        Parameters
        ----------
        other : TimeseriesItem
            Another TimeseriesItem to compare with

        Returns
        -------
        bool
            True if items are equal, False otherwise
        """
        if not isinstance(other, TimeseriesItem):
            return False
        names_are_equal = self.name == other.name
        values_are_equal = np.allclose(self.values, other.values)
        time_is_equal = all(self.time == other.time)
        attrs_are_equal = self.attrs == other.attrs

        return (
            names_are_equal and values_are_equal and time_is_equal and attrs_are_equal
        )

    def __hash__(self):
        """Return hash of the item based on its name.

        Returns
        -------
        int
            Hash value
        """
        return hash(self.name)

"""Time series data container and operations."""

from __future__ import annotations

import os

os.environ["DARTS_CONFIGURE_MATPLOTLIB"] = "0"

from collections.abc import Iterable
from typing import Dict, List, Optional, Self, Union

import mikeio
import numpy as np
import pandas as pd
import xarray as xr
from darts.timeseries import (  # type: ignore[import-untyped]
    TimeSeries as DartsTimeSeries,
)

from blue_ml.timeseries._mixins import (
    _BaseFunctionsMixin,
    _DataPropertiesMixin,
    _TSDataFormatMapperMixin,
)
from blue_ml.timeseries.timeseries_item import ItemAlias, ItemClass, TimeseriesItem


class ItemList:
    """
    Helper class to store multiple timeseries items and provide access to them.

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing the items.
    """

    def __init__(self, data: xr.Dataset):
        self._items = {
            name: TimeseriesItem(data[name], str(name)) for name in data.data_vars
        }
        # Preserve the time coordinate for empty datasets
        self._time_coord = data.coords.get("time")

    def add(self, data: xr.Dataset):
        """
        Add new items to the collection.

        Parameters
        ----------
        data : xr.Dataset
            Dataset containing new items to add.
        """
        new_items = {
            name: TimeseriesItem(data[name], str(name)) for name in data.data_vars
        }
        self._items.update(new_items)

    def to_dataset(self, subset: Optional[Iterable] = None) -> xr.Dataset:
        """
        Convert items to xarray Dataset.

        Parameters
        ----------
        subset : iterable or None, optional
            Subset of item names to include. If None, includes all items.
            Default is None.

        Returns
        -------
        xr.Dataset
            Dataset containing the selected items.
        """
        if subset is None:
            subset = list(self._items.keys())
        items_as_ds = {
            key: value._ds for key, value in self._items.items() if key in subset
        }
        ds = xr.Dataset(items_as_ds)
        # Preserve time coordinate for empty datasets
        if not items_as_ds and self._time_coord is not None:
            ds = ds.assign_coords(time=self._time_coord)
        return ds

    def __getitem__(self, key: str) -> TimeseriesItem:
        return self._items[key]

    def __repr__(self):
        return f"ItemList({[name for name in self._items.keys()]})"


class TimeseriesData(
    _BaseFunctionsMixin, _TSDataFormatMapperMixin, _DataPropertiesMixin
):
    """
    Class for subset of timeseries data (targets or features).

    This class provides a container for either features or targets in a timeseries,
    with utilities for manipulation and format conversion.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series or xr.DataArray or xr.Dataset or mikeio.DataArray
        Input timeseries data.
    item_class : ItemClass
        Class of items (FEATURE or TARGET).
    """

    _empty_ds = xr.Dataset(coords={"time": ("time", pd.to_datetime([]))})

    def __init__(
        self,
        data: Union[
            pd.DataFrame, pd.Series, xr.DataArray, xr.Dataset, mikeio.DataArray
        ],
        item_class: ItemClass,
    ):
        self._item_class = item_class
        ds = self._to_ds_with_empty_fallback(data)
        # Store the time coordinate to preserve it even when there are no data variables
        self._time_coord = ds.coords.get("time")
        self.items = ItemList(ds)

    def _to_ds_with_empty_fallback(
        self,
        data: Union[
            pd.DataFrame, pd.Series, xr.DataArray, xr.Dataset, mikeio.DataArray
        ],
    ) -> xr.Dataset:
        """Convert input data to Dataset, allowing empty timeless xarray inputs."""
        try:
            ds = self._type_to_ds(data)
        except ValueError:
            if self._is_empty_xarray_without_time(data):
                return self._empty_ds.copy()
            raise
        return ds

    @staticmethod
    def _is_empty_xarray_without_time(
        data: Union[
            pd.DataFrame, pd.Series, xr.DataArray, xr.Dataset, mikeio.DataArray
        ],
    ) -> bool:
        if isinstance(data, xr.Dataset):
            return "time" not in data.dims and len(data.data_vars) == 0
        if isinstance(data, xr.DataArray):
            return "time" not in data.dims and data.size == 0
        return False

    def contains_item(self, name: str) -> bool:
        """
        Check if an item with the given name exists.

        Parameters
        ----------
        name : str
            Name of the item to check.

        Returns
        -------
        bool
            True if item exists, False otherwise.
        """
        return name in self.names

    def add_items(
        self,
        data: Union[
            np.ndarray,
            pd.DataFrame,
            pd.Series,
            xr.DataArray,
            xr.Dataset,
        ],
    ):
        """
        Add new items to the timeseries data.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame or pd.Series or xr.DataArray or xr.Dataset
            Data to add as new items.
        """
        data = self._type_to_ds(data)
        # _type_to_ds should always return xr.Dataset
        assert isinstance(data, xr.Dataset), "Expected _type_to_ds to return xr.Dataset"
        self.items.add(data)

    def __repr__(self):
        return (
            self.as_xarray()
            .__repr__()
            .replace("xarray.Dataset", f"TimeseriesData.{self.item_class.name}View")
        )

    def _repr_html_(self):
        return (
            self.as_xarray()
            ._repr_html_()
            .replace("xarray.Dataset", f"TimeseriesData.{self.item_class.name}View")
        )

    def __getitem__(self, key: Union[str, Iterable]) -> Union[TimeseriesItem, Self]:
        if isinstance(key, str):
            return self.items[key]
        elif isinstance(key, Iterable):
            new_ds = self.items.to_dataset(subset=key)
            return type(self)(new_ds, self.item_class)

    def __setitem__(self, item: str, other: Union[float, Iterable, TimeseriesItem]):
        target_item = self.items[item]
        if isinstance(other, TimeseriesItem):
            target_item._ds.values = other.values
            # TODO: add attrs
        elif isinstance(other, float | int):
            target_item._ds.values = np.repeat(other, self.n_time)
        elif isinstance(other, Iterable):
            # Convert to list to ensure it has __len__
            from typing import Sized, cast

            if not hasattr(other, "__len__"):
                other_sized: Sized = list(other)
            else:
                other_sized = cast(Sized, other)
            if len(other_sized) == self.n_time:
                target_item._ds.values = other_sized
            else:
                raise ValueError(
                    f"Lengths ({self.n_time}, {len(other_sized)})do not match"
                )
        else:
            raise ValueError(
                "Values can only be set passing another 'TimeseriesItem' or an array"
            )

    def __iter__(self):
        self._pointer = 0
        return self

    def __next__(self):
        self._pointer += 1
        if self._pointer < (len(self.names) + 1):
            return self.to_dataframe().iloc[:, self._pointer - 1]
        raise StopIteration

    def to_dataset(self) -> xr.Dataset:
        return self.as_xarray()

    def drop(self, names: Union[List[str], str]):
        ds_after_drop = self.as_xarray().drop_vars(names)
        return type(self)(ds_after_drop, self.item_class)

    def rename(self, renamer_dict: Dict[str, str]) -> Self:
        # TODO: update attribute names

        renamed_ds = self.as_xarray().rename_vars(
            {name: renamer_dict.get(name, name) for name in self.names}
        )
        return type(self)(renamed_ds, self.item_class)

    @property
    def names(self) -> List[str]:
        return [str(key) for key in self.items._items.keys()]

    @property
    def item_class(self) -> ItemClass:
        return self._item_class

    def update_item_attrs(self, new_attrs: dict):
        for name, item_attrs in new_attrs.items():
            item = self[name]
            if isinstance(item, TimeseriesItem):
                item.update_attrs(item_attrs)
            else:
                raise ValueError(
                    f"Cannot update attributes for item '{name}' - not a TimeseriesItem"
                )

    @property
    def attrs(self) -> dict[str, dict]:
        return {
            name: self[name].attrs for name in self.names if len(self[name].attrs) > 0
        }

    def as_xarray(self) -> xr.Dataset:
        ds = self.items.to_dataset()
        # If the dataset has no data variables but we have a stored time coordinate,
        # add it back to preserve time information for empty TimeseriesData
        if len(ds.data_vars) == 0 and self._time_coord is not None:
            ds = ds.assign_coords(time=self._time_coord)
        return ds

    def copy_item_attributes(
        self,
        from_item: ItemAlias,
        to_item: ItemAlias,
        unchanged: Optional[Union[str, List[str]]] = None,
    ):
        """Copy the attributes from an item in the timeseries to a different item.

        Parameters
        ----------
        from_item : ItemAlias
            Item to copy the attributes from
        to_item : ItemAlias
            Target item to copy the attributes to
        unchanged : Optional[Union[str, List[str]]], optional
            attributes that should not be changed in the target item, by default ["name"]
        """
        if unchanged is None:
            unchanged = ["name"]
        elif isinstance(unchanged, str):
            unchanged = [unchanged]

        original_attributes = self[to_item.name].attrs
        new_attributes = self[
            from_item.name
        ].attrs.copy()  # Make a copy to avoid modifying original
        for attr in unchanged:
            if attr in original_attributes:
                new_attributes[attr] = original_attributes[attr]

        self.update_item_attrs({to_item.name: new_attributes})

    def intersect_names(self, names: str | List[str]) -> List[str]:
        """Return the subset of names that is present in TimeseriesData object.

        Parameters
        ----------
        names : str | List[str]
            Names that might be in object

        Returns
        -------
        List[str]
            Subset of names
        """
        if isinstance(names, str):
            names = [names]
        return list(set(self.names).intersection(set(names)))

    def get_item_alias(self, item_name: str) -> ItemAlias:
        return ItemAlias.from_variable_name(name=item_name)

    def to_darts(self) -> DartsTimeSeries:
        """Convert to Darts TimeSeries.

        Returns
        -------
        DartsTimeSeries
            Converted Darts time series
        """
        from blue_ml.timeseries.converters import FormatConverter

        return FormatConverter.to_darts(self.as_xarray())

    def gaps(self, mode: str = "all") -> pd.DataFrame:
        # Cast mode to the expected literal type
        from typing import Literal

        if mode not in ["all", "any"]:
            raise ValueError(f"mode must be 'all' or 'any', got {mode}")
        mode_literal: Literal["all", "any"] = mode  # type: ignore[assignment]
        return self.to_darts().gaps(mode=mode_literal)

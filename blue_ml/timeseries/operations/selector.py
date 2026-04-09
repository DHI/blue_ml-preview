"""Selection operations for Timeseries.

This module handles selection operations like sel, isel, and attribute-based selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional, Sequence, Union

import xarray as xr

if TYPE_CHECKING:
    from blue_ml.timeseries.timeseries import Timeseries


class TimeseriesSelector:
    """Handles selection operations for Timeseries.

    Responsibility: Select subsets of timeseries data based on various criteria.
    """

    @staticmethod
    def sel(
        ts: Timeseries,
        indexers=None,
        method: Optional[str] = None,
        tolerance: Optional[Union[int, float, Sequence[Union[int, float]]]] = None,
        drop: bool = False,
        **indexers_kwargs,
    ) -> Timeseries:
        """Select data using label-based indexing.

        Parameters
        ----------
        ts : Timeseries
            Timeseries to select from
        indexers : dict, optional
            Indexers to select
        method : str, optional
            Method to use for inexact matches
        tolerance : int, float, or sequence, optional
            Maximum distance for inexact matches
        drop : bool, default=False
            Drop coordinates not matching the selection
        **indexers_kwargs
            Additional indexers

        Returns
        -------
        Timeseries
            New Timeseries with selected data
        """
        from blue_ml.timeseries.timeseries import Timeseries

        ds = ts.as_xarray().sel(
            indexers=indexers,
            method=method,
            tolerance=tolerance,
            drop=drop,
            **indexers_kwargs,
        )

        # Handle empty targets case
        if not ts.has_targets:
            empty_targets = xr.Dataset(coords={"time": ds.coords["time"]})
            return Timeseries(
                features=ds[ts.features.names],
                targets=empty_targets,
                item_attrs=ts.attrs,
            )

        return Timeseries(
            features=ds[ts.features.names],
            targets=ds[ts.targets.names],
            item_attrs=ts.attrs,
        )

    @staticmethod
    def isel(
        ts: Timeseries,
        indexers=None,
        drop: bool = False,
        missing_dims: Literal["raise", "warn", "ignore"] = "raise",
        **indexers_kwargs,
    ) -> Timeseries:
        """Select data using integer-based indexing.

        Parameters
        ----------
        ts : Timeseries
            Timeseries to select from
        indexers : dict, optional
            Integer indexers to select
        drop : bool, default=False
            Drop coordinates not matching the selection
        missing_dims : {'raise', 'warn', 'ignore'}, default='raise'
            How to handle missing dimensions
        **indexers_kwargs
            Additional indexers

        Returns
        -------
        Timeseries
            New Timeseries with selected data
        """
        from blue_ml.timeseries.timeseries import Timeseries

        ds = ts.as_xarray().isel(
            indexers=indexers, drop=drop, missing_dims=missing_dims, **indexers_kwargs
        )

        # Handle empty targets case
        if not ts.has_targets:
            empty_targets = xr.Dataset(coords={"time": ds.coords["time"]})
            return Timeseries(
                features=ds[ts.features.names],
                targets=empty_targets,
                item_attrs=ts.attrs,
            )

        return Timeseries(
            features=ds[ts.features.names],
            targets=ds[ts.targets.names],
            item_attrs=ts.attrs,
        )

    @staticmethod
    def sel_from_attrs(ts: Timeseries, **attr_equals) -> Timeseries:
        """Select items based on their attributes.

        Parameters
        ----------
        ts : Timeseries
            Timeseries to select from
        **attr_equals
            Attribute key-value pairs to match

        Returns
        -------
        Timeseries
            New Timeseries with items matching the attributes
        """
        from blue_ml.timeseries.timeseries import Timeseries
        from blue_ml.timeseries.preprocessing._parser import find_names_by_attr

        name_list = find_names_by_attr(ts.as_xarray(), attr_equals)
        # Filter names into features and targets
        feature_names = [name for name in name_list if name in ts.features.names]
        target_names = [name for name in name_list if name in ts.targets.names]
        item_attrs = {name: ts.attrs[name] for name in name_list if name in ts.attrs}

        # Create new Timeseries with selected names
        return Timeseries(
            data=ts.as_xarray(),
            features=feature_names,
            targets=target_names,
            item_attrs=item_attrs,
        )

    @staticmethod
    def get_item_names_from_attrs(ts: Timeseries, **attr_equals) -> List[str]:
        """Get item names matching the specified attributes.

        Parameters
        ----------
        ts : Timeseries
            Timeseries to search
        **attr_equals
            Attribute key-value pairs to match

        Returns
        -------
        list of str
            Names of items matching the attributes
        """
        from blue_ml.timeseries.preprocessing._parser import find_names_by_attr

        return find_names_by_attr(ts.as_xarray(), attr_equals)

"""Modification operations for Timeseries.

This module handles adding, dropping, and renaming items in Timeseries objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Union

import mikeio
import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from blue_ml.timeseries.timeseries import Timeseries
    from blue_ml.timeseries.timeseries_item import ItemClass


class TimeseriesModifier:
    """Handles modification operations for Timeseries.

    Responsibility: Add, drop, and rename items in timeseries.
    """

    @staticmethod
    def add_items(
        ts: Timeseries,
        data: Union[
            np.ndarray,
            pd.DataFrame,
            pd.Series,
            xr.DataArray,
            xr.Dataset,
            mikeio.DataArray,
            mikeio.Dataset,
        ],
        item_class: ItemClass,
        names: Optional[List[str]] = None,
        time: Optional[pd.DatetimeIndex] = None,
        attrs: Optional[Dict] = None,
        overwrite: bool = False,
    ) -> Timeseries:
        """Add items to timeseries.

        Parameters
        ----------
        ts : Timeseries
            Timeseries to add items to
        data : various
            Data to add
        item_class : ItemClass
            Whether adding features or targets
        names : list of str, optional
            Names for the items
        time : pd.DatetimeIndex, optional
            Time index for the data
        attrs : dict, optional
            Attributes for the items
        overwrite : bool, default=False
            Whether to overwrite existing items

        Returns
        -------
        Timeseries
            New Timeseries with items added
        """
        from blue_ml.timeseries.converters import FormatConverter
        from blue_ml.timeseries.timeseries_item import ItemClass

        data_ds = FormatConverter.to_dataset(data, names=names, time=time)

        # Extract attrs from data itself
        data_attrs = {}
        for var in data_ds.data_vars:
            if data_ds[var].attrs:
                data_attrs[str(var)] = dict(data_ds[var].attrs)

        # Merge with provided attrs (provided attrs take precedence)
        if attrs:
            for key, value in attrs.items():
                if key in data_attrs:
                    # Merge: attrs from argument override data attrs
                    data_attrs[key].update(value)
                else:
                    data_attrs[key] = value

        # Check for overlaps with opposite category (features vs targets)
        if item_class == ItemClass.FEATURE:
            overlapping = set(data_ds.data_vars).intersection(set(ts.targets.names))
            if overlapping:
                raise ValueError(
                    f"Passed data contains '{overlapping}' that match target names. Rename before adding."
                )
            # Check for overlaps within same category if not overwriting
            if not overwrite:
                existing_overlaps = set(data_ds.data_vars).intersection(
                    set(ts.features.names)
                )
                if existing_overlaps:
                    raise ValueError(
                        f"Feature(s) '{existing_overlaps}' already exist. Set overwrite=True to replace them."
                    )
        else:  # TARGET
            overlapping = set(data_ds.data_vars).intersection(set(ts.features.names))
            if overlapping:
                raise ValueError(
                    f"Passed data contains '{overlapping}' that match feature names. Rename before adding."
                )
            # Check for overlaps within same category if not overwriting
            if not overwrite:
                existing_overlaps = set(data_ds.data_vars).intersection(
                    set(ts.targets.names)
                )
                if existing_overlaps:
                    raise ValueError(
                        f"Target(s) '{existing_overlaps}' already exist. Set overwrite=True to replace them."
                    )

        # Create new timeseries with added items
        ts_copy = ts.copy()
        if item_class == ItemClass.FEATURE:
            ts_copy.features.add_items(data_ds)
        else:
            ts_copy.targets.add_items(data_ds)

        # Apply merged attributes
        if data_attrs:
            ts_copy.set_item_attrs(data_attrs)

        return ts_copy

    @staticmethod
    def drop_items(
        ts: Timeseries,
        vars: Union[str, List[str]],
        allow_empty_targets: bool = False,
    ) -> Timeseries:
        """Drop items from timeseries.

        Parameters
        ----------
        ts : Timeseries
            Timeseries to drop items from
        vars : str or list of str
            Names of items to drop
        allow_empty_targets : bool, default=False
            Whether to allow dropping all targets

        Returns
        -------
        Timeseries
            New Timeseries with items dropped

        Raises
        ------
        ValueError
            If trying to drop all features or all targets (when not allowed)
        """
        from blue_ml.timeseries.timeseries import Timeseries

        if isinstance(vars, str):
            vars = [vars]

        features_to_keep = [f for f in ts.features.names if f not in vars]
        targets_to_keep = [t for t in ts.targets.names if t not in vars]

        if len(features_to_keep) == 0:
            raise ValueError(
                "You are removing all features, there should be at least one left."
            )
        if len(targets_to_keep) == 0 and not allow_empty_targets:
            raise ValueError(
                "You are removing all targets, there should be at least one left."
            )

        # Filter attributes for kept items
        attrs = {
            k: v for k, v in ts.attrs.items() if k in features_to_keep + targets_to_keep
        }

        return Timeseries(
            ts.as_xarray(),
            features=features_to_keep,
            targets=targets_to_keep,
            item_attrs=attrs,
        )

    @staticmethod
    def rename_items(
        ts: Timeseries,
        renamer_dict: Dict[str, str],
    ) -> Timeseries:
        """Rename items in timeseries.

        Parameters
        ----------
        ts : Timeseries
            Timeseries to rename items in
        renamer_dict : dict[str, str]
            Mapping of old names to new names

        Returns
        -------
        Timeseries
            New Timeseries with items renamed
        """
        from blue_ml.timeseries.timeseries import Timeseries

        renamer_features = {
            k: v for k, v in renamer_dict.items() if k in ts.features.names
        }
        renamer_targets = {
            k: v for k, v in renamer_dict.items() if k in ts.targets.names
        }

        renamed_features = ts.features.rename(renamer_features)
        renamed_targets = ts.targets.rename(renamer_targets)

        return Timeseries(
            features=renamed_features,
            targets=renamed_targets,
            item_attrs=ts.attrs,
        )

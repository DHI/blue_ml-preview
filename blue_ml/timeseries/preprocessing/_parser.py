"""Helper methods for parsing and preparing data for Timeseries initialization."""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import mikeio
import pandas as pd
import xarray as xr

from blue_ml.dataprocessing.utils import match_attrs_wildcard
from blue_ml.timeseries.timeseries_data import TimeseriesData


def find_names_by_attr(
    ds: xr.Dataset | xr.DataArray, attrs: dict[str, str]
) -> List[str]:
    """Select variable names from a dataset whose attributes match the given key-value pairs.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Dataset or DataArray to search for matching variables.
    attrs : dict of str to str
        Attribute key-value pairs to match against. Variables whose attributes contain
        all specified key-value pairs are included. Wildcard patterns in keys are
        supported via ``match_attrs_wildcard``.

    Returns
    -------
    list of str
        Names of variables in ``ds`` whose attributes match all entries in ``attrs``.
    """
    # Prepare variable names for wildcard expansion depending on input type
    if isinstance(ds, xr.Dataset):
        var_names = [str(v) for v in ds.data_vars]
    else:
        # xr.DataArray: treat the array itself as a single variable
        var_names = [str(ds.name)] if ds.name is not None else []

    attrs = match_attrs_wildcard(attrs, var_names)

    items: List[str] = []

    if isinstance(ds, xr.Dataset):
        # Select variables whose attributes match all key/value pairs
        for varname in ds.data_vars:
            var_attrs = ds[varname].attrs
            if all(
                (attr_key in var_attrs and var_attrs[attr_key] == attr_value)
                for attr_key, attr_value in attrs.items()
            ):
                items.append(str(varname))
    else:
        # xr.DataArray: check attributes on the array itself
        if var_names:
            var_attrs = ds.attrs
            if all(
                (attr_key in var_attrs and var_attrs[attr_key] == attr_value)
                for attr_key, attr_value in attrs.items()
            ):
                items.append(var_names[0])
    return items


def get_item_names(
    data: Optional[xr.Dataset],
    match: Optional[Union[str, List[str], Dict[str, Any]]],
) -> List[str]:
    """Retrieve variable names from a dataset based on a name list or attribute filter.

    Parameters
    ----------
    data : xr.Dataset or None
        Dataset to search. Must not be ``None``.
    match : str, list of str, dict of str to any, or None
        Selector for which variables to return.
        - ``None``: returns an empty list.
        - ``str``: wrapped into a single-element list.
        - ``list of str``: returned as-is.
        - ``dict``: treated as attribute filters passed to ``find_names_by_attr``.

    Returns
    -------
    list of str
        Variable names selected from ``data`` according to ``match``.

    Raises
    ------
    ValueError
        If ``data`` is ``None``.
    """
    if data is None:
        raise ValueError("'data' should not be empty.")

    if match is None:
        return []
    elif isinstance(match, dict):
        match = find_names_by_attr(data, match)
    elif isinstance(match, str):
        match = [match]

    return match


def prepare_item_attrs(
    attrs: Dict[str, Any], other_attrs: Dict[str, Any], names: List[str]
) -> Dict[str, Any]:
    """Merge and normalise per-item attribute dictionaries.

    Resolves wildcard keys in ``attrs``, ensures attribute definitions are
    consistently structured per item, and merges ``other_attrs`` into the result.

    Parameters
    ----------
    attrs : dict of str to any
        Primary attribute mapping. Keys may be item names (applying attributes
        to specific items), arbitrary attribute keys (applying to all items),
        or wildcard patterns that are expanded against ``names``.
    other_attrs : dict of str to any
        Secondary attribute mapping to merge into ``attrs``. Shared keys are
        merged at the per-item level; remaining entries are added directly.
    names : list of str
        Canonical list of item names used for wildcard expansion and validation.

    Returns
    -------
    dict of str to any
        Merged attribute dictionary keyed by item name, where each value is
        a dict of attributes for that item.

    Raises
    ------
    ValueError
        If ``attrs`` contains some—but not all—item names as keys, which would
        result in ambiguous per-item vs. global attribute assignment.
    """
    contains_wildcards = any("*" in k for k in attrs.keys())
    if contains_wildcards:
        attrs = match_attrs_wildcard(attrs, names)

    # We check if the passed attributes are defined by item or apply to all
    all_keys_are_names = set(attrs.keys()).issubset(set(names))
    some_keys_are_names = len(set(attrs.keys()).intersection(set(names))) > 0
    if some_keys_are_names and (not all_keys_are_names):
        raise ValueError(
            "'item_attrs' should be specified per item (including wildcards) or without any item as key."
        )

    if not all_keys_are_names:  # The attrs will be applied to all items
        attrs = {key: attrs for key in names}

    shared_keys = set(other_attrs.keys()).intersection(attrs.keys())
    if len(shared_keys) > 0:
        for var in shared_keys:
            attrs[var].update(other_attrs.pop(var))
    attrs.update(other_attrs)

    return attrs


def transform_data_and_get_attrs(
    tsdata: Union[
        TimeseriesData,
        mikeio.DataArray,
        mikeio.Dataset,
        pd.DataFrame,
        pd.Series,
        xr.Dataset,
        xr.DataArray,
    ],
) -> Tuple[pd.DataFrame, Dict]:
    """Convert a data object to a DataFrame and extract per-variable attributes.

    Parameters
    ----------
    tsdata : TimeseriesData, mikeio.DataArray, mikeio.Dataset, pd.DataFrame,
             pd.Series, xr.Dataset, or xr.DataArray
        Input data to convert. If already a ``pd.DataFrame``, it is returned
        unchanged. Otherwise it is converted via ``to_dataframe()`` / ``to_frame()``.

    Returns
    -------
    pd.DataFrame
        Tabular representation of ``tsdata``.
    dict
        Per-variable attribute dictionary. Keys are variable (column) names;
        values are dicts of attributes extracted from the source object's
        metadata (e.g. ``xr.DataArray.attrs`` or ``mikeio`` item metadata).
        Empty dict if no attributes are found or ``tsdata`` is a plain
        ``pd.DataFrame``.
    """
    # TODO: improve so we do not need to convert to DataFrame
    # Currently, TimeseriesData objects cannot be initialized passing another
    # TimeseriesData object, so we transform to df

    attrs_by_item: Dict = {}
    if not isinstance(tsdata, pd.DataFrame):
        try:
            # Check if it's an xarray Dataset with data_vars
            if isinstance(tsdata, xr.Dataset):
                attrs_by_item = {
                    str(var): tsdata[var].attrs for var in tsdata.data_vars
                }
            elif isinstance(tsdata, xr.DataArray):
                attrs_by_item = {
                    tsdata.name: tsdata.attrs,
                }
            elif isinstance(tsdata, mikeio.Dataset):
                for itm in tsdata.items:
                    attrs_by_item[itm.name] = {
                        "name": itm.name,
                        "units": itm.unit.name,
                        "eumType": itm.type,
                        "eumUnit": itm.unit,
                    }
            elif isinstance(tsdata, mikeio.DataArray):
                attrs_by_item[tsdata.name] = {
                    "name": tsdata.name,
                    "units": tsdata.unit.name,
                    "eumType": tsdata.type,
                    "eumUnit": tsdata.unit,
                }

        except (AttributeError, TypeError):
            pass
        # Note: We don't extract global attrs here as they should be
        # handled separately and not mixed with per-item attrs
        if isinstance(tsdata, pd.Series):
            tsdata = tsdata.to_frame()
        else:
            # Use type ignore for the diverse types that all have to_dataframe
            tsdata = tsdata.to_dataframe()  # type: ignore[attr-defined]
    return tsdata, attrs_by_item


def extract_features_and_targets(
    data: Optional[xr.Dataset],
    features: Optional[
        Union[
            str,
            List[str],
            Dict[str, str],
            TimeseriesData,
            mikeio.DataArray,
            mikeio.Dataset,
            pd.DataFrame,
            pd.Series,
            xr.Dataset,
            xr.DataArray,
        ]
    ],
    targets: Optional[
        Union[
            str,
            List[str],
            Dict[str, str],
            TimeseriesData,
            mikeio.DataArray,
            mikeio.Dataset,
            pd.DataFrame,
            pd.Series,
            xr.Dataset,
            xr.DataArray,
        ]
    ],
) -> Tuple[
    Union[pd.DataFrame, xr.Dataset],
    Union[pd.DataFrame, xr.Dataset],
    Dict[str, Dict],
    Dict[str, Any],
]:
    """Split a data source into separate feature and target DataFrames or Datasets.

    Supports two initialisation modes:

    1. **Data + selectors** — ``data`` is an ``xr.Dataset`` and both ``features``
       and ``targets`` are name selectors (``str``, ``list of str``, or ``dict``
       of attribute filters). Variables are selected directly from ``data``.
    2. **Independent objects** — ``data`` is ``None`` and ``features``/``targets``
       are data objects (``pd.DataFrame``, ``pd.Series``, ``xr.Dataset``,
       ``xr.DataArray``, ``mikeio.Dataset``, ``mikeio.DataArray``, or
       ``TimeseriesData``). Each is converted to a DataFrame independently.

    Parameters
    ----------
    data : xr.Dataset or None
        Combined dataset from which features and targets are selected by name or
        attribute. Pass ``None`` when providing ``features`` and ``targets`` as
        independent data objects.
    features : str, list of str, dict of str to str, or data object, or None
        Feature specification. When ``data`` is provided, used as a name/attribute
        selector. Otherwise, a data object converted to a DataFrame.
    targets : str, list of str, dict of str to str, or data object, or None
        Target specification. Same semantics as ``features``.

    Returns
    -------
    features_result : pd.DataFrame or xr.Dataset
        Feature data extracted or converted from the input.
    targets_result : pd.DataFrame or xr.Dataset
        Target data extracted or converted from the input.
    attrs_by_item : dict of str to dict
        Per-variable attribute dictionaries for all feature and target variables.
    global_attrs : dict of str to any
        Dataset-level (global) attributes, if available.

    Raises
    ------
    ValueError
        If the combination of ``data``, ``features``, and ``targets`` does not
        match either supported initialisation mode, or if requested variable
        names are not found in ``data``.
    """
    by_item_types = (dict, str, list)
    by_data_types = (
        pd.DataFrame,
        pd.Series,
        TimeseriesData,
        mikeio.DataArray,
        mikeio.Dataset,
        xr.DataArray,
        xr.Dataset,
    )

    global_attrs: Dict[str, Any] = {}

    data_is_passed = (
        (data is not None)
        and isinstance(features, by_item_types)
        and isinstance(targets, by_item_types)
    )
    features_and_targets_only = (
        (data is None)
        and isinstance(features, by_data_types)
        and isinstance(targets, by_data_types)
    )

    if data_is_passed:
        # Type narrowing: we know data is not None and features/targets are by_item_types here
        assert data is not None, "data should not be None in data_is_passed branch"
        assert isinstance(features, (dict, str, list)), "Type narrowing failed"
        assert isinstance(targets, (dict, str, list)), "Type narrowing failed"

        feature_names = get_item_names(data, features)
        target_names = get_item_names(data, targets)

        all_names = feature_names + target_names
        all_names_in_data = all([name in data.data_vars for name in all_names])

        if all_names_in_data:
            features_result = data[feature_names]
            targets_result = data[target_names]
        else:
            raise ValueError("Not all names found in data")

        global_attrs.update(data.attrs)
        attrs_by_item = {var: data[var].attrs for var in all_names}

        return features_result, targets_result, attrs_by_item, global_attrs

    elif features_and_targets_only:
        # Type narrowing: we know features and targets are by_data_types here
        assert isinstance(features, by_data_types), "Type narrowing failed"
        assert isinstance(targets, by_data_types), "Type narrowing failed"

        try:
            # Safe attribute access with hasattr check
            features_attrs = getattr(features, "attrs", {})
            targets_attrs = getattr(targets, "attrs", {})
            if features_attrs and targets_attrs:
                global_attrs.update(features_attrs | targets_attrs)
        except (AttributeError, TypeError):
            pass

        features_df, feature_attrs_by_item = transform_data_and_get_attrs(features)  # type: ignore[arg-type]
        targets_df, target_attrs_by_item = transform_data_and_get_attrs(targets)  # type: ignore[arg-type]

        feature_names = list(features_df.columns)
        target_names = list(targets_df.columns)

        all_names = feature_names + target_names
        attrs_by_item = feature_attrs_by_item | target_attrs_by_item

        return features_df, targets_df, attrs_by_item, global_attrs

    else:
        raise ValueError(
            f"Timeseries can only be initialized in two ways: passing 'data', and 'features'/'targets' as one of {by_item_types} or 'features'/'targets' as {by_data_types} and 'data' as None"
        )

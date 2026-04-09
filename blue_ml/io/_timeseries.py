"""Miscellaneous utility functions for the io module."""

import xarray as xr


def write_ds(ds: xr.Dataset, filename, features, targets, **kwargs):
    """Write a xarray dataset to a file.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to write
    filename : str
        Output filepath
    """
    # We need to add the _vartype attribute to the data variables
    # in order for the .read() function to work properly
    # We save  target/feature tags as a _vartype attribute
    ds = ds.copy()

    for name in features:
        ds[name].attrs.update({"_vartype": "feature"})
    for name in targets:
        ds[name].attrs.update({"_vartype": "target"})

    # We need to convert boolean attributes to strings
    attrs_vars = {v: ds[v].attrs for v in ds.data_vars}
    attrs_vars = replace_bool_with_str(attrs_vars)
    for v in ds.data_vars:
        ds[v].attrs = attrs_vars[v]

    if ".nc" not in filename:
        filename = filename + ".nc"

    # Write
    ds.to_netcdf(filename, **kwargs)


def read_ds(filename, force_unspecified=None, drop_tags=True, **kwargs):
    """Read a xarray dataset from a file.

    Parameters
    ----------
    filename : str
        File to read from

    Returns
    -------
    xarray.Dataset
        Dataset read from file
    """
    ds = xr.open_dataset(filename, **kwargs)

    # Reverse string attribures to boolean attributes
    attrs_vars = {v: ds[v].attrs for v in ds.data_vars}
    attrs_vars = replace_str_with_bool(attrs_vars)
    for v in ds.data_vars:
        ds[v].attrs = attrs_vars[v]

    # Read feature/target tags
    # Remove tags from data
    # Drop tag, optional
    def drop_tag(
        name,
    ):
        if drop_tags:
            ds[name].attrs.pop("_vartype")

    features = []
    targets = []

    for name in ds.data_vars:
        if ds[name].attrs.get("_vartype") == "feature":
            drop_tag(
                name,
            )
            features.append(name)
        elif ds[name].attrs.get("_vartype") == "target":
            drop_tag(name)
            targets.append(name)
        else:
            if force_unspecified is not None:
                if force_unspecified == "feature":
                    features.append(name)
                elif force_unspecified == "target":
                    targets.append(name)
            else:
                raise ValueError(
                    f"'_vartype' not found in {name}.attrs. Unknown target feature type for {name}. Try force_unspecified='feature' or 'target'"
                )

    return ds, features, targets


def replace_bool_with_str(d):
    """
    Recursively replace boolean values with strings in a dictionary.

    Parameters
    ----------
    d : dict
        Dictionary potentially containing boolean values.

    Returns
    -------
    dict
        Dictionary with booleans converted to strings.
    """
    d = d.copy()
    for k, v in d.items():
        if isinstance(v, bool):
            d[k] = str(v)
        if isinstance(v, dict):
            d[k] = replace_bool_with_str(v)
    return d


def replace_str_with_bool(d):
    """
    Recursively replace string 'true'/'false' with boolean values in a dictionary.

    Parameters
    ----------
    d : dict
        Dictionary potentially containing 'true'/'false' strings.

    Returns
    -------
    dict
        Dictionary with string booleans converted to bool type.
    """
    d = d.copy()
    for k, v in d.items():
        if isinstance(v, str) and v.lower() in ["true", "false"]:
            d[k] = bool(v)
        if isinstance(v, dict):
            d[k] = replace_str_with_bool(v)
    return d

"""I/O operations for reading and writing Timeseries data.

This module handles all file I/O operations for Timeseries objects,
following the Single Responsibility Principle by separating I/O concerns
from the core Timeseries data structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union

import xarray as xr

if TYPE_CHECKING:
    from blue_ml.timeseries.timeseries import Timeseries


class TimeseriesReader:
    """Handles reading Timeseries from files.

    Responsibility: Read Timeseries data from various file formats.
    """

    @staticmethod
    def read(
        filename: str,
        force_unspecified: Union[None, Literal["target", "feature"]] = None,
        **xr_kwargs,
    ) -> xr.Dataset:
        """Read timeseries dataset from a netCDF file.

        Will infer target/feature from tags as a _vartype attribute.

        Parameters
        ----------
        filename : str
            Filepath to read from
        force_unspecified : None | 'target' | 'feature'
            If 'feature' or 'target', will force all unspecified variables to be of that type
        **xr_kwargs : key-value pairs
            Additional keyword arguments to be passed to xarray.open_dataset

        Returns
        -------
        xr.Dataset
            Dataset containing the timeseries data

        Raises
        ------
        ValueError
            If force_unspecified is not 'target' or 'feature'
        RuntimeError
            If file cannot be read with any available engine
        """
        try:
            ds = xr.open_dataset(filename, **xr_kwargs)
        except (RuntimeError, OSError) as e:
            # Handle NETCDF3 files that may have been written with scipy backend
            # The netCDF4 backend may complain about endian encoding in NETCDF3 files
            if "only endian='native' allowed for NETCDF3 files" in str(e):
                ds = xr.open_dataset(filename, engine="scipy", **xr_kwargs)
            # Handle HDF errors - fall back to h5netcdf
            elif "HDF error" in str(e):
                ds = xr.open_dataset(filename, engine="h5netcdf", **xr_kwargs)
            else:
                raise

        # Validate force_unspecified parameter
        if force_unspecified is not None and force_unspecified not in [
            "target",
            "feature",
        ]:
            raise ValueError("force_unspecified must be 'target' or 'feature'")

        return ds


class TimeseriesWriter:
    """Handles writing Timeseries to files.

    Responsibility: Write Timeseries data to various file formats.
    """

    @staticmethod
    def write(
        ts: Timeseries,
        filename: str,
        **kwargs,
    ) -> None:
        """Write Timeseries contents to a netCDF file.

        Will save target/feature tags as a _vartype attribute for reading later.

        Parameters
        ----------
        ts : Timeseries
            Timeseries object to write
        filename : str
            Path to which to save this dataset
        **kwargs : key-value pairs
            Additional keyword arguments to be passed to xarray.Dataset.to_netcdf
        """
        from blue_ml.timeseries.timeseries_item import ItemClass

        # Get the xarray dataset representation
        ds = ts.as_xarray()

        # Add _vartype attribute to distinguish features and targets
        for name in ts.features.names:
            ds[name].attrs.update({"_vartype": ItemClass.FEATURE.name.lower()})
        for name in ts.targets.names:
            ds[name].attrs.update({"_vartype": ItemClass.TARGET.name.lower()})

        # Convert any dict-type attributes to strings for netCDF compatibility
        for var in ds.data_vars:
            for attr_key, attr_val in ds[var].attrs.items():
                if isinstance(attr_val, dict):
                    ds[var].attrs[attr_key] = str(attr_val)
        for attr_key, attr_val in ds.attrs.items():
            if isinstance(attr_val, dict):
                ds.attrs[attr_key] = str(attr_val)

        # Ensure .nc extension
        if not filename.endswith(".nc"):
            filename = filename + ".nc"

        try:
            ds.to_netcdf(filename, **kwargs)
        except RuntimeError:
            # Fallback to h5netcdf if default engine fails
            ds.to_netcdf(filename, engine="h5netcdf", **kwargs)

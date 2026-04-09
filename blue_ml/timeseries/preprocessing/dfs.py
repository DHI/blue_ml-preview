"""DFS0 file processing and dataset management."""

import os
import re
from typing import Callable, List, Optional, Tuple

import mikeio


class Dfs0Dataset:
    """Dataset reader for MIKE DFS0 files.

    Parameters
    ----------
    filename : str
        Name of the DFS0 file
    filedir : str
        Directory containing the DFS0 file
    varname_generator : Optional[Callable], optional
        Function to generate variable names from filename, by default None
    """

    def __init__(
        self, filename: str, filedir: str, varname_generator: Optional[Callable] = None
    ):
        self.filename = filename
        self._filepath = filedir + filename
        if varname_generator is not None:
            self.alias = varname_generator(filename)
        else:
            self.alias = filename

        self.data: Optional[mikeio.Dataset] = None
        self.coords: Optional[Tuple[float, float]] = None

    def read(self):
        """Read the DFS0 file and extract coordinates from filename."""
        self.data = mikeio.read(self._filepath)
        self.coords = self.find_coords(self.filename)  # (Longitude, Latitude)

    def data_as_array(self):
        """Convert the DFS0 data to an xarray DataArray.

        Returns
        -------
        xr.DataArray
            Data as xarray DataArray

        Raises
        ------
        AssertionError
            If no file has been read yet
        """
        assert self.data is not None, "No file has been read"

        data_xr = self.data.to_xarray().to_dataarray()
        data_xr.name = self.alias

        return data_xr

    @staticmethod
    def find_coords(dfs0_filename: str) -> Tuple[float, float]:
        """Find coordinates included in a dfs0 file name.

        The function assumes that the coordinates are the first two consecutive float numbers
        (separated by an "_") that are found on the title. The order is (longitude, latitude).

        Parameters
        ----------
        input_string : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        parts = dfs0_filename.split("_")
        # Regex pattern to match floats
        float_pattern = re.compile(r"^-?\d*\.\d+$")
        lon_lat = []
        for i in range(len(parts) - 1):
            if float_pattern.match(parts[i]) and float_pattern.match(parts[i + 1]):
                lon_lat.append(float(parts[i]))
                lon_lat.append(float(parts[i + 1]))
                break

        if len(lon_lat) != 2:
            raise ValueError(
                f"Could not find exactly two coordinates in filename: {dfs0_filename}"
            )

        return (lon_lat[0], lon_lat[1])


class DatasetCollection:
    """Collection of DFS0 datasets with utilities for batch processing.

    Parameters
    ----------
    varname_generator : Optional[Callable], optional
        Function to generate variable names from filenames, by default None
    """

    def __init__(self, varname_generator: Optional[Callable] = None):
        self.varname_generator = varname_generator
        self.datasets: List[Dfs0Dataset] = []

    def read_from_directory(
        self,
        datapath: os.PathLike,
    ):
        """Read all DFS0 files from a directory.

        Parameters
        ----------
        datapath : os.PathLike
            Path to directory containing DFS0 files
        """
        dfs0_filenames = [fn for fn in os.listdir(datapath) if fn[-4:] == "dfs0"]

        mood_datasets = []
        for filename in dfs0_filenames:
            mdi = Dfs0Dataset(
                filename=filename,
                filedir=str(datapath),
                varname_generator=self.varname_generator,
            )
            mdi.read()
            mood_datasets.append(mdi)

        self.datasets = mood_datasets

    def read_from_list(
        self,
        list_of_datasets: List[Dfs0Dataset],
    ):
        """Load datasets from a list of Dfs0Dataset objects.

        Parameters
        ----------
        list_of_datasets : List[Dfs0Dataset]
            List of pre-created Dfs0Dataset objects
        """
        self.datasets = list_of_datasets

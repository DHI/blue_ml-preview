"""Core Timeseries class implementation."""

from __future__ import annotations

import os

os.environ["DARTS_CONFIGURE_MATPLOTLIB"] = "0"

import sys
from typing import (
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

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
from blue_ml.timeseries._report import TimeseriesReport
from blue_ml.timeseries.plotting._plot import _Plotter
from blue_ml.timeseries.preprocessing import DatasetCollection, Dfs0Dataset
from blue_ml.timeseries.preprocessing._parser import (
    extract_features_and_targets,
    prepare_item_attrs,
)
from blue_ml.timeseries.timeseries_data import TimeseriesData
from blue_ml.timeseries.timeseries_item import (
    ItemAlias,
    ItemClass,
    TimeseriesItem,
)

if sys.version_info < (3, 11):  # Self is not available in typing module before 3.11
    from typing_extensions import Self
else:
    from typing import Self


def _parse_item_class(item_class: Union[str, ItemClass]) -> ItemClass:
    target_aliases = ["target", "targets"]
    feature_aliases = ["feature", "features"]

    if isinstance(item_class, ItemClass):
        return item_class
    else:
        if item_class.lower() in target_aliases:
            return ItemClass.TARGET
        elif item_class.lower() in feature_aliases:
            return ItemClass.FEATURE
        else:
            raise ValueError("Invalid item class: try `target` or `feature`")


class Timeseries(_BaseFunctionsMixin, _TSDataFormatMapperMixin, _DataPropertiesMixin):
    """Xarray based timeseries data structure with features and targets.

    Parameters
    ----------
    data : Timeseries | xr.Dataset, optional
        Direct initialization from Timeseries-like class , by default None
    features : mikeio.DataArray | pd.DataFrame | xr.Dataset, optional
        Data for features, by default None
    targets : mikeio.DataArray | pd.DataFrame | xr.Dataset, optional
        Data for Targets, by default None
    attrs : dict, optional
        Attributes for the dataset, by default None
    global_attrs : dict, optional
        Global attributes for the dataset, by default None

    Examples
    --------
    >>> np.random.seed(0)
    >>> features = np.random.randn(10, 3)
    >>> targets = np.random.rand(10, 1)
    >>> reference_time = pd.date_range(start='1/1/2018', periods=10, freq='D')
    >>> df_features=pd.DataFrame(features, index=reference_time, columns = ['a', 'b', 'c'])
    >>> df_targets=pd.DataFrame(targets, index=reference_time, columns = ['z'])
    >>> ts = Timeseries(features=df_features, targets=df_targets)
    >>> ts
    <blue_ml.Timeseries> Size: 400B
    Dimensions:  (time: 10)
    Coordinates:
    * time     (time) datetime64[ns] 80B 2018-01-01 2018-01-02 ... 2018-01-10
    Features:
        a        (time) float64 80B 1.764 2.241 0.9501 ... 0.6536 2.27 -0.1872
        b        (time) float64 80B 0.4002 1.868 -0.1514 ... 0.8644 -1.454 1.533
        c        (time) float64 80B 0.9787 -0.9773 -0.1032 ... -0.7422 0.04576 1.469
    Targets:
        z        (time) float64 80B 0.9437 0.6818 0.3595 ... 0.6706 0.2104 0.1289

    """

    plotter = _Plotter
    transform = None
    _join_method: Literal["outer", "inner", "left", "right", "exact", "override"] = (
        "outer"
    )

    def __init__(
        self,
        data: Optional[xr.Dataset] = None,
        *,
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
        ] = None,
        item_attrs: Optional[Dict] = None,
        global_attrs: Optional[Dict] = None,
    ):
        features, targets, _item_attrs, _global_attrs = extract_features_and_targets(
            data, features, targets
        )

        # At this point, features and targets are valid data types for TimeseriesData
        self.features = TimeseriesData(features, ItemClass.FEATURE)  # type: ignore[arg-type]
        self.targets = TimeseriesData(targets, ItemClass.TARGET)  # type: ignore[arg-type]

        # Check for time alignment between features and targets
        self._check_time_alignment()

        if global_attrs is None:
            global_attrs = {}
        global_attrs.update(_global_attrs)
        self.global_attrs = global_attrs.copy()

        if item_attrs is None:
            item_attrs = {}
        item_attrs = prepare_item_attrs(item_attrs, _item_attrs, self.names)

        # Initialize attribute manager with SOLID principles
        from blue_ml.timeseries.attributes import AttributeManager

        self.attr_manager = AttributeManager(item_attrs, global_attrs)

        # Apply the prepared attributes to the actual items
        self.set_item_attrs(item_attrs)

        # Initialize renderer for display
        from blue_ml.timeseries.rendering import TimeseriesRenderer

        self._renderer = TimeseriesRenderer()

        self.report = TimeseriesReport(self)
        self.plot = Timeseries.plotter(self)

    def _check_time_alignment(self) -> None:
        """Check if features and targets have aligned time axes and warn if not.

        Issues a warning when:
        - Features and targets have different time axes
        - There is partial or no overlap between feature and target timesteps

        This helps users identify potential issues when using misaligned data for ML.
        """
        import warnings

        features_time = set(self.features.time)
        targets_time = set(self.targets.time)

        if features_time != targets_time:
            overlap = features_time.intersection(targets_time)
            n_features = len(features_time)
            n_targets = len(targets_time)
            n_overlap = len(overlap)

            if n_overlap == 0:
                warnings.warn(
                    f"Features and targets have NO overlapping timesteps. "
                    f"Features: {n_features} timesteps, Targets: {n_targets} timesteps. "
                    f"This data cannot be used for supervised learning without alignment.",
                    UserWarning,
                    stacklevel=3,
                )
            else:
                warnings.warn(
                    f"Features and targets have different time axes. "
                    f"Features: {n_features} timesteps, Targets: {n_targets} timesteps, "
                    f"Overlap: {n_overlap} timesteps. "
                    f"Only {n_overlap} samples will be valid for supervised learning.",
                    UserWarning,
                    stacklevel=3,
                )

    def _get_single_item(self, item: str) -> TimeseriesItem:
        try:
            result = self.features[item]
            # When accessing with a string key, features[item] should return TimeseriesItem
            assert isinstance(
                result, TimeseriesItem
            ), f"Expected TimeseriesItem, got {type(result)}"
            return result
        except KeyError:
            pass
        try:
            result = self.targets[item]
            # When accessing with a string key, targets[item] should return TimeseriesItem
            assert isinstance(
                result, TimeseriesItem
            ), f"Expected TimeseriesItem, got {type(result)}"
            return result
        except KeyError:
            pass
        raise KeyError(f"'{item}' not found in Timeseries")

    @overload
    def __getitem__(self, key: str) -> TimeseriesItem: ...

    @overload
    def __getitem__(self, key: Iterable) -> Union[TimeseriesData, TimeseriesItem]: ...

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._get_single_item(key)
        else:
            try:
                features = self.features[key]  # All selected_names are features
            except KeyError:
                features = None

            try:
                targets = self.targets[key]  # All selected_names are targets
            except KeyError:
                targets = None

            if (targets is not None) and (features is None):
                return targets
            elif (targets is None) and (features is not None):
                return features
            elif (targets is not None) and (features is not None):
                # Both features and targets are found - ensure they are TimeseriesData
                from blue_ml.timeseries.timeseries_data import TimeseriesData

                assert isinstance(
                    features, TimeseriesData
                ), f"Expected TimeseriesData for features, got {type(features)}"
                assert isinstance(
                    targets, TimeseriesData
                ), f"Expected TimeseriesData for targets, got {type(targets)}"
                return type(self)(
                    features=features,
                    targets=targets,
                    item_attrs=self.attrs,
                )
            else:
                raise KeyError(
                    "None of the passed variables are not found in Timeseries"
                )

    def __iter__(self):
        self._pointer = 0
        return self

    def __next__(self):
        self._pointer += 1
        if self._pointer < (len(self.names) + 1):
            return self.to_dataframe().iloc[:, self._pointer - 1]
        raise StopIteration

    def __repr__(self) -> str:
        """String representation using TimeseriesRenderer."""
        return self._renderer.to_string(self)

    def _repr_html_(self) -> str:
        """HTML representation using TimeseriesRenderer."""
        return self._renderer.to_html(self)

    @property
    def is_valid_for_ml(self) -> bool:
        """Check if the dataset is valid for ML use.

        Delegates to MLAdapter for SRP compliance.
        See `self.report` for details

        Returns
        -------
        bool
            True if valid, False otherwise
        """
        from blue_ml.timeseries.ml import MLAdapter

        return MLAdapter.is_valid_for_ml(self)

    @classmethod
    def read(
        cls, filename: str, force_unspecified: Union[None, str] = None, **xr_kwargs
    ) -> Timeseries:
        """Open and decode a Timeseries from a netcdf file or file-like object.

        Will infer target/feature from tags as a _vartype attribute.

        Parameters
        ----------
        filename : str
            Filepath to read from
        force_unspecified : None | str
            If 'feature' or 'target', will force all unspecified variables to be of that type
        **kwargs : key-value pairs
            Additional keyword arguments to be passed to the reader
            See `xarray.open_dataset`

        Returns
        -------
        Timeseries
            Timeseries object
        """
        from blue_ml.timeseries.io import TimeseriesReader

        # Validate force_unspecified parameter
        if force_unspecified is not None and force_unspecified not in [
            "target",
            "feature",
        ]:
            raise ValueError("force_unspecified must be 'target' or 'feature'")

        # Default to 'feature' if None and cast to appropriate literal type
        if force_unspecified is None:
            validated_force_unspecified: Literal["target", "feature"] = "feature"
        else:
            validated_force_unspecified = force_unspecified  # type: ignore[assignment]

        ds = TimeseriesReader.read(filename, validated_force_unspecified, **xr_kwargs)
        return cls.from_dataset(ds, force_unspecified=validated_force_unspecified)

    def write(self, filename: str, **kwargs) -> None:
        """Write Timeseries contents to a netCDF file.

        Will save target/feature tags as a _vartype attribute for reading later.

        Parameters
        ----------
        filename : str
            Path to which to save this dataset
        **kwargs : key-value pairs
            Additional keyword arguments to be passed to the writer
            See `xarray.Dataset.to_netcdf`
        """
        from blue_ml.timeseries.io import TimeseriesWriter

        TimeseriesWriter.write(self, filename, **kwargs)

    def resample(
        self,
        rule: str,
        method: Literal[
            "mean", "sum", "max", "min", "median", "std", "first", "last"
        ] = "mean",
    ) -> Self:
        """Resample timeseries to a different frequency.

        Delegates to TimeseriesResampler for SRP compliance.
        """
        from blue_ml.timeseries.operations import TimeseriesResampler

        result = TimeseriesResampler.resample(self, rule, method)
        return cast(Self, result)

    def contains_item(
        self,
        name: str,
        item_class: Optional[Union[ItemClass, Literal["target", "feature"]]] = None,
    ) -> bool:
        """
        Check if an item with the given name exists in the timeseries.

        Parameters
        ----------
        name : str
            Name of the item to check.
        item_class : ItemClass or {'target', 'feature'} or None, optional
            Limit search to specific item class. If None, search all items.
            Default is None.

        Returns
        -------
        bool
            True if item exists, False otherwise.
        """
        if item_class is None:
            return name in self.names
        else:
            item_class = _parse_item_class(item_class)
            if item_class == ItemClass.FEATURE:
                return self.features.contains_item(name)
            elif item_class == ItemClass.TARGET:
                return self.targets.contains_item(name)

    @classmethod
    def from_index_and_values(
        cls,
        values: np.ndarray,
        index: Union[pd.Index, pd.DatetimeIndex],
        target_names: Union[str, List[str]],
        column_names: List[str],
        attrs: Optional[Dict] = None,
    ) -> Self:
        """
        Create Timeseries from array values and index.

        Parameters
        ----------
        values : np.ndarray
            2D array of values with shape (time, columns).
        index : pd.Index or pd.DatetimeIndex
            Time index for the data.
        target_names : str or list of str
            Names of target variables (must be in column_names).
        column_names : list of str
            Names for all columns in values array.
        attrs : dict or None, optional
            Attributes for the items. Default is None.

        Returns
        -------
        Timeseries
            New Timeseries object.

        Raises
        ------
        ValueError
            If target_names is not a string or list of strings.
        """
        if isinstance(target_names, str):
            target_names = [target_names]
        if not all([isinstance(s, str) for s in target_names]):
            raise ValueError(
                "'target_names' should be passed as a string or list of strings."
            )
        data = xr.DataArray(
            values, coords=[index, column_names], dims=["time", "columns"]
        ).to_dataset("columns")
        for name in target_names:
            column_names.remove(name)
        return cls(
            features=data[column_names], targets=data[target_names], item_attrs=attrs
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        target_names: Union[str, List[str]],
        attrs: Optional[Dict] = None,
    ) -> Self:
        """
        Create Timeseries from pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with time index and feature/target columns.
        target_names : str or list of str
            Names of target columns.
        attrs : dict or None, optional
            Attributes for the items. Default is None.

        Returns
        -------
        Timeseries
            New Timeseries object.
        """
        return cls.from_index_and_values(
            values=df.values,
            index=df.index,
            target_names=target_names,
            column_names=df.columns.to_list(),
            attrs=attrs,
        )

    @classmethod
    def from_dataset(
        cls,
        ds: xr.Dataset,
        force_unspecified: Literal["target", "feature"],
        drop_tags: bool = True,
    ) -> Self:
        """Convert xarray.Dataset to Timeseries.

        Will infer target/feature from tags as a _vartype attribute.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to convert
        features : List of strings
            Names of data_vars to be considered as features
        targets : List of strings
            Names of data_vars to be considered as targets

        Returns
        -------
        Timeseries
            Timeseries object
        """

        def classify_item_names(which: ItemClass):
            all_names = ds.data_vars
            item_class = which.name.lower()
            names_of_matching_class = [
                name
                for name in all_names
                if ds[name].attrs.get("_vartype") == item_class
            ]
            # Drop attrs if specified, to avoid confusion after classification
            if drop_tags:
                for name in names_of_matching_class:
                    ds[name].attrs.pop("_vartype")

            other_names = list(set(all_names).difference(names_of_matching_class))

            return names_of_matching_class, other_names

        target_names, non_target_names = classify_item_names(ItemClass.TARGET)
        feature_names, non_feature_names = classify_item_names(ItemClass.FEATURE)

        if force_unspecified is not None:
            parsed_force_unspecified = _parse_item_class(force_unspecified)
            unspecified_names = list(
                set(non_target_names).intersection(non_feature_names)
            )
            if parsed_force_unspecified == ItemClass.FEATURE:
                feature_names += unspecified_names
            elif parsed_force_unspecified == ItemClass.TARGET:
                target_names += unspecified_names
            else:
                raise ValueError("Unsupported item class")

        return cls(features=ds[feature_names], targets=ds[target_names])

    @classmethod
    def from_darts(
        cls,
        ts: DartsTimeSeries,
        feature_names: List[str],
        target_names: List[str],
        attrs: Optional[Dict] = None,
    ) -> Self:
        ds = ts._xa.to_dataset("component").isel({"sample": 0})
        return cls(
            features=ds[feature_names], targets=ds[target_names], item_attrs=attrs
        )

    @classmethod
    def from_dataset_collection(
        cls,
        collection: DatasetCollection,
        target_filename: str,
        target_variable: str,
    ) -> Self:
        """Initializes a Timeseries object from a collection of dfs0 files.

        The files should be compiled in a DatasetCollection object. Also, it is
        necessary to define which file from the collection contains the target
        variable and which is the target variable. All global variables are treated as features.

        Parameters
        ----------
        collection : DatasetCollection
            Collection of dfs0 files
        target_filename : str
            Name of the dfs0 file containing the target variable
        target_variable : str
            Name of the target variable

        Returns
        -------
        Self
            Timeseries object with defined target
        """

        # TODO: expand for multiple target variables
        def write_alias(filename: str):
            if collection.varname_generator is None:
                return filename
            else:
                return collection.varname_generator(filename)

        def renamer_dict(md: Dfs0Dataset) -> Dict:
            if md.data is None:
                raise ValueError(
                    f"Dataset {md.filename} has not been read yet. Call .read() first."
                )
            variable_names = md.data.names
            item_names = [
                ItemAlias(varname, write_alias(md.filename)).name
                for varname in variable_names
            ]
            return {key: value for key, value in zip(variable_names, item_names)}

        def set_target_dataset(
            datasets: List[Dfs0Dataset],
        ) -> Tuple[Dfs0Dataset, List[Dfs0Dataset]]:
            is_target = [md.filename == target_filename for md in datasets]

            assert sum(is_target) == 1, "There needs to be exactly one target dataset"
            idx_ref = np.where(is_target)[0][0]

            target_ds = datasets[idx_ref]
            other_ds = [ds for i, ds in enumerate(datasets) if i != idx_ref]

            if target_ds.data is None:
                raise ValueError(
                    f"Target dataset {target_ds.filename} has not been read yet. Call .read() first."
                )
            target_ds.data = target_ds.data[[target_variable]]
            target_ds.data = target_ds.data.rename(renamer_dict(target_ds))

            return target_ds, other_ds

        def assemble_timeseries(
            target_ds: Dfs0Dataset, global_ds: List[Dfs0Dataset]
        ) -> Timeseries:
            if global_ds[0].data is None:
                raise ValueError(
                    f"Dataset {global_ds[0].filename} has not been read yet. Call .read() first."
                )
            ts = cls(
                features=global_ds[0].data.rename(renamer_dict(global_ds[0])),
                targets=target_ds.data,
            )
            for md in global_ds[1:]:
                if md.data is None:
                    raise ValueError(
                        f"Dataset {md.filename} has not been read yet. Call .read() first."
                    )
                ts.add_features(md.data.rename(renamer_dict(md)))
            return ts

        target_ds, other_ds = set_target_dataset(collection.datasets)
        ts = assemble_timeseries(target_ds, other_ds)

        # Type checkers expect Self, but we're returning the same class type
        from typing import cast

        return cast(Self, ts)

    def set_item_attrs(self, new_attrs: Dict[str, dict]) -> None:
        """Set metadata for items in Timeseries.

        Delegates to AttributeManager for SRP compliance.

        Parameters
        ----------
        new_attrs : dict[str, dict]
            Dictionary mapping item names to their attributes
        """
        # Update both the attribute manager and the actual data structures
        self.attr_manager.update_all_item_attrs(new_attrs)

        # Also update the actual TimeseriesData objects
        feature_attrs = {k: v for k, v in new_attrs.items() if k in self.features.names}
        self.features.update_item_attrs(feature_attrs)

        target_attrs = {k: v for k, v in new_attrs.items() if k in self.targets.names}
        self.targets.update_item_attrs(target_attrs)

    @property
    def attrs(self) -> dict[str, dict]:
        """Get all item attributes.

        Returns combined attributes from features and targets.
        """
        return self.features.attrs | self.targets.attrs

    def as_xarray(self) -> xr.Dataset:
        features = self.features.to_dataset()
        targets = self.targets.to_dataset()

        return xr.merge([features, targets], join="outer", compat="no_conflicts")

    @property
    def names(self) -> List[str]:
        return self.features.names + self.targets.names

    def fill_missing_values(
        self, fill: Union[str, float] = "auto", **interpolate_kwargs
    ):
        """Fill missing values in the timeseries.

        Delegates to MissingValueHandler for SRP compliance.
        """
        from blue_ml.timeseries.operations import MissingValueHandler

        return MissingValueHandler.fill_missing_values(self, fill, **interpolate_kwargs)

    def gaps(self, mode="all") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Detect gaps in features and targets.

        Delegates to MissingValueHandler for SRP compliance.
        """
        from blue_ml.timeseries.operations import MissingValueHandler

        return MissingValueHandler.detect_gaps(self, mode)

    def dropna(self, how: Literal["any", "all"] = "any", thresh: "int | None" = None):
        """Drop missing values.

        Delegates to MissingValueHandler for SRP compliance.
        """
        from blue_ml.timeseries.operations import MissingValueHandler

        return MissingValueHandler.dropna(self, how, thresh)

    def add_features(
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
        attrs: Optional[Dict] = None,
        overwrite: bool = False,
    ):
        """Add features to the timeseries.

        Delegates to TimeseriesModifier for SRP compliance.

        Parameters
        ----------
        data : various
            Data to add as features
        names : list of str, optional
            Names for the new features
        time : pd.DatetimeIndex, optional
            Time index for the data
        attrs : dict, optional
            Attributes for the new features
        overwrite : bool, default=False
            Whether to overwrite existing features
        """
        from blue_ml.timeseries.operations import TimeseriesModifier

        result = TimeseriesModifier.add_items(
            self, data, ItemClass.FEATURE, names, time, attrs, overwrite
        )
        # Update self for consistency with mutable pattern
        self.features = result.features
        self.targets = result.targets

    def add_targets(
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
        attrs: Optional[Dict] = None,
        overwrite: bool = False,
    ):
        """Add targets to the timeseries.

        Delegates to TimeseriesModifier for SRP compliance.

        Parameters
        ----------
        data : various
            Data to add as targets
        names : list of str, optional
            Names for the new targets
        time : pd.DatetimeIndex, optional
            Time index for the data
        attrs : dict, optional
            Attributes for the new targets
        overwrite : bool, default=False
            Whether to overwrite existing targets
        """
        from blue_ml.timeseries.operations import TimeseriesModifier

        result = TimeseriesModifier.add_items(
            self, data, ItemClass.TARGET, names, time, attrs, overwrite
        )
        # Update self for consistency with mutable pattern
        self.features = result.features
        self.targets = result.targets

    def rename(
        self, renamer_dict: Dict[str, str], inplace: bool = False
    ) -> None | Timeseries:
        """Rename items in the timeseries.

        Delegates to TimeseriesModifier for SRP compliance.

        Parameters
        ----------
        renamer_dict : dict[str, str]
            Mapping of old names to new names
        inplace : bool, default=False
            Whether to rename in place or return a new Timeseries

        Returns
        -------
        Timeseries or None
            New Timeseries if inplace=False, None otherwise
        """
        from blue_ml.timeseries.operations import TimeseriesModifier

        result = TimeseriesModifier.rename_items(self, renamer_dict)

        if inplace:
            self.features = result.features
            self.targets = result.targets
            return None
        else:
            return result

    @overload
    def drop(
        self, vars: Union[str, List[str]], inplace: Literal[False] = False
    ) -> Timeseries: ...

    @overload
    def drop(self, vars: Union[str, List[str]], inplace: Literal[True]) -> None: ...

    def drop(
        self, vars: Union[str, List[str]], inplace: bool = False
    ) -> Optional[Timeseries]:
        """Drop items from the timeseries.

        Delegates to TimeseriesModifier for SRP compliance.

        Parameters
        ----------
        vars : str or list of str
            Names of items to drop
        inplace : bool, default=False
            Whether to drop in place or return a new Timeseries

        Returns
        -------
        Timeseries or None
            New Timeseries if inplace=False, None otherwise
        """
        from blue_ml.timeseries.operations import TimeseriesModifier

        # Check if we're allowing empty targets (for prediction scenarios)
        allow_empty_targets = not self.has_targets

        result = TimeseriesModifier.drop_items(self, vars, allow_empty_targets)

        if inplace:
            self.features = result.features
            self.targets = result.targets
            return None
        else:
            return result

    def sel(
        self,
        indexers=None,
        method: Optional[str] = None,
        tolerance: Optional[Union[int, float, Sequence[Union[int, float]]]] = None,
        drop: bool = False,
        **indexers_kwargs,
    ) -> Timeseries:
        """Select data using label-based indexing.

        Delegates to TimeseriesSelector for SRP compliance.
        """
        from blue_ml.timeseries.operations import TimeseriesSelector

        return TimeseriesSelector.sel(
            self, indexers, method, tolerance, drop, **indexers_kwargs
        )

    def isel(
        self,
        indexers=None,
        drop: bool = False,
        missing_dims: Literal["raise", "warn", "ignore"] = "raise",
        **indexers_kwargs,
    ) -> Timeseries:
        """Select data using integer-based indexing.

        Delegates to TimeseriesSelector for SRP compliance.
        """
        from blue_ml.timeseries.operations import TimeseriesSelector

        return TimeseriesSelector.isel(
            self, indexers, drop, missing_dims, **indexers_kwargs
        )

    @property
    def n_features(self) -> int:
        return len(self.features.names)

    @property
    def n_targets(self) -> int:
        return len(self.targets.names)

    @property
    def has_targets(self) -> bool:
        """Check if timeseries has meaningful targets.

        Returns
        -------
        bool
            True if targets exist and are not empty
        """
        return not self.targets.is_empty and len(self.targets.names) > 0

    def sel_from_attrs(self, **attr_equals) -> Timeseries:
        """Select items based on their attributes.

        Delegates to TimeseriesSelector for SRP compliance.
        """
        from blue_ml.timeseries.operations import TimeseriesSelector

        return TimeseriesSelector.sel_from_attrs(self, **attr_equals)

    def get_item_names_from_attrs(self, **attr_equals) -> List[str]:
        """Get item names matching the specified attributes.

        Delegates to TimeseriesSelector for SRP compliance.
        """
        from blue_ml.timeseries.operations import TimeseriesSelector

        return TimeseriesSelector.get_item_names_from_attrs(self, **attr_equals)

    def set_multihorizon_target(self, horizon: int) -> Timeseries:
        """Set up multihorizon targets for forecasting.

        Delegates to MLAdapter for SRP compliance.
        """
        from blue_ml.timeseries.ml import MLAdapter

        return MLAdapter.set_multihorizon_target(self, horizon)

    def __setitem__(self, item: str, other: Union[float, Iterable, TimeseriesItem]):
        if isinstance(other, TimeseriesItem):
            self[item]._ds.values = other.values
            # TODO: add attrs
        elif isinstance(other, float | int):
            self[item]._ds.values = np.repeat(other, self.n_time)
        elif isinstance(other, Iterable):
            # Convert to list to ensure it has __len__
            from typing import Sized, cast

            if not hasattr(other, "__len__"):
                other_sized: Sized = list(other)
            else:
                other_sized = cast(Sized, other)
            if len(other_sized) == self.n_time:
                self[item]._ds.values = other_sized
            else:
                raise ValueError(
                    f"Lengths ({self.n_time}, {len(other_sized)})do not match"
                )
        else:
            raise ValueError(
                "Values can only be set passing another 'TimeseriesItem' or an array"
            )

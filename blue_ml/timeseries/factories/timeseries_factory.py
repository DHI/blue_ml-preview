"""Factory for creating Timeseries objects.

This module provides factory methods for creating Timeseries from various data sources.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

import mikeio
import numpy as np
import pandas as pd
import xarray as xr
from darts.timeseries import TimeSeries as DartsTimeSeries  # type: ignore[import-untyped]

from blue_ml.timeseries.timeseries_data import TimeseriesData


class TimeseriesFactory:
    """Factory for creating Timeseries objects from various sources.

    Responsibility: Construct Timeseries objects from different data formats and sources.
    """

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        target_names: Union[str, List[str]],
        attrs: Optional[Dict] = None,
    ):
        """Create Timeseries from pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with time index and feature/target columns
        target_names : str or list of str
            Names of target columns
        attrs : dict, optional
            Attributes for the items

        Returns
        -------
        Timeseries
            New Timeseries object
        """
        from blue_ml.timeseries.timeseries import Timeseries

        return Timeseries.from_index_and_values(
            values=df.values,
            index=df.index,
            target_names=target_names,
            column_names=df.columns.to_list(),
            attrs=attrs,
        )

    @staticmethod
    def from_index_and_values(
        values: np.ndarray,
        index: Union[pd.Index, pd.DatetimeIndex],
        target_names: Union[str, List[str]],
        column_names: List[str],
        attrs: Optional[Dict] = None,
    ):
        """Create Timeseries from array values and index.

        Parameters
        ----------
        values : np.ndarray
            2D array of values with shape (time, columns)
        index : pd.Index or pd.DatetimeIndex
            Time index for the data
        target_names : str or list of str
            Names of target variables (must be in column_names)
        column_names : list of str
            Names for all columns in values array
        attrs : dict, optional
            Attributes for the items

        Returns
        -------
        Timeseries
            New Timeseries object

        Raises
        ------
        ValueError
            If target_names is not a string or list of strings
        """
        from blue_ml.timeseries.timeseries import Timeseries

        if isinstance(target_names, str):
            target_names = [target_names]
        if not all([isinstance(s, str) for s in target_names]):
            raise ValueError(
                "'target_names' should be passed as a string or list of strings."
            )

        data = xr.DataArray(
            values, coords=[index, column_names], dims=["time", "columns"]
        ).to_dataset("columns")

        feature_names = [name for name in column_names if name not in target_names]

        return Timeseries(
            features=data[feature_names], targets=data[target_names], item_attrs=attrs
        )

    @staticmethod
    def from_dataset(
        ds: xr.Dataset,
        force_unspecified: Literal["target", "feature"] = "feature",
        drop_tags: bool = True,
    ):
        """Convert xarray.Dataset to Timeseries.

        Will infer target/feature from tags as a _vartype attribute.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to convert
        force_unspecified : {'target', 'feature'}, default='feature'
            How to classify variables without _vartype attribute
        drop_tags : bool, default=True
            Whether to drop _vartype tags after reading

        Returns
        -------
        Timeseries
            Timeseries object
        """
        from blue_ml.timeseries.timeseries import Timeseries, _parse_item_class
        from blue_ml.timeseries.timeseries_item import ItemClass

        def classify_item_names(which: ItemClass):
            all_names = ds.data_vars
            item_class = which.name.lower()
            names_of_matching_class = [
                name
                for name in all_names
                if ds[name].attrs.get("_vartype") == item_class
            ]
            matching_names = []
            for name in names_of_matching_class:
                if drop_tags:
                    ds[name].attrs.pop("_vartype")
                matching_names.append(name)

            other_names = list(set(all_names).difference(matching_names))

            return matching_names, other_names

        target_names, non_target_names = classify_item_names(ItemClass.TARGET)
        feature_names, non_feature_names = classify_item_names(ItemClass.FEATURE)

        parsed_force_unspecified = _parse_item_class(force_unspecified)
        unspecified_names = list(
            set(non_target_names).symmetric_difference(non_feature_names)
        )
        if parsed_force_unspecified == ItemClass.FEATURE:
            feature_names += unspecified_names
        elif parsed_force_unspecified == ItemClass.TARGET:
            target_names += unspecified_names

        return Timeseries(features=ds[feature_names], targets=ds[target_names])

    @staticmethod
    def from_darts(
        ts: DartsTimeSeries,
        feature_names: List[str],
        target_names: List[str],
        attrs: Optional[Dict] = None,
    ):
        """Create Timeseries from Darts TimeSeries.

        Parameters
        ----------
        ts : DartsTimeSeries
            Darts TimeSeries object
        feature_names : list of str
            Names of feature variables
        target_names : list of str
            Names of target variables
        attrs : dict, optional
            Attributes for the items

        Returns
        -------
        Timeseries
            New Timeseries object
        """
        from blue_ml.timeseries.timeseries import Timeseries

        ds = ts._xa.to_dataset("component").isel({"sample": 0})
        return Timeseries(
            features=ds[feature_names], targets=ds[target_names], item_attrs=attrs
        )

    @staticmethod
    def for_training(
        features: Union[
            mikeio.DataArray,
            mikeio.Dataset,
            pd.DataFrame,
            pd.Series,
            xr.Dataset,
            xr.DataArray,
            TimeseriesData,
        ],
        targets: Union[
            mikeio.DataArray,
            mikeio.Dataset,
            pd.DataFrame,
            pd.Series,
            xr.Dataset,
            xr.DataArray,
            TimeseriesData,
        ],
        item_attrs: Optional[Dict] = None,
        global_attrs: Optional[Dict] = None,
    ):
        """Create Timeseries for model training (with both features and targets).

        This factory method makes it explicit that you're creating a timeseries
        for model training, where both features and targets are required.

        Parameters
        ----------
        features : mikeio.DataArray | mikeio.Dataset | pd.DataFrame | pd.Series | xr.Dataset | xr.DataArray | TimeseriesData
            Feature data
        targets : mikeio.DataArray | mikeio.Dataset | pd.DataFrame | pd.Series | xr.Dataset | xr.DataArray | TimeseriesData
            Target data
        item_attrs : dict, optional
            Attributes for individual items
        global_attrs : dict, optional
            Global attributes for the entire timeseries

        Returns
        -------
        Timeseries
            Timeseries configured for training with both features and targets

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from blue_ml.timeseries.factories import TimeseriesFactory
        >>>
        >>> # Create training data
        >>> time = pd.date_range('2020-01-01', periods=100, freq='H')
        >>> features_df = pd.DataFrame({
        ...     'temp': np.random.randn(100),
        ...     'pressure': np.random.randn(100)
        ... }, index=time)
        >>> targets_df = pd.DataFrame({
        ...     'output': np.random.randn(100)
        ... }, index=time)
        >>>
        >>> # Create timeseries for training
        >>> ts = TimeseriesFactory.for_training(
        ...     features=features_df,
        ...     targets=targets_df
        ... )
        >>> assert ts.has_targets is True
        """
        from blue_ml.timeseries.timeseries import Timeseries

        return Timeseries(
            features=features,
            targets=targets,
            item_attrs=item_attrs,
            global_attrs=global_attrs,
        )

    @staticmethod
    def for_prediction(
        features: Union[
            mikeio.DataArray,
            mikeio.Dataset,
            pd.DataFrame,
            pd.Series,
            xr.Dataset,
            xr.DataArray,
            TimeseriesData,
        ],
        item_attrs: Optional[Dict] = None,
        global_attrs: Optional[Dict] = None,
    ):
        """Create Timeseries for prediction (features only, no targets needed).

        This factory method makes it explicit that you're creating a timeseries
        for making predictions in production, where only features are available
        and no target values exist yet.

        Parameters
        ----------
        features : mikeio.DataArray | mikeio.Dataset | pd.DataFrame | pd.Series | xr.Dataset | xr.DataArray | TimeseriesData
            Feature data
        item_attrs : dict, optional
            Attributes for individual items
        global_attrs : dict, optional
            Global attributes for the entire timeseries

        Returns
        -------
        Timeseries
            Timeseries configured for prediction (features only, empty targets)

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from blue_ml.timeseries.factories import TimeseriesFactory
        >>>
        >>> # Create new data for prediction (no targets)
        >>> time = pd.date_range('2020-06-01', periods=24, freq='H')
        >>> features_df = pd.DataFrame({
        ...     'temp': np.random.randn(24),
        ...     'pressure': np.random.randn(24)
        ... }, index=time)
        >>>
        >>> # Create timeseries for prediction only
        >>> ts = TimeseriesFactory.for_prediction(features=features_df)
        >>> assert ts.has_targets is False
        """
        from blue_ml.timeseries.timeseries import Timeseries

        # Create empty targets based on features type
        empty_targets: Union[pd.DataFrame, xr.Dataset]
        if isinstance(features, pd.DataFrame):
            empty_targets = pd.DataFrame(index=features.index)
        elif isinstance(features, pd.Series):
            empty_targets = pd.DataFrame(index=features.index)
        elif isinstance(features, (xr.Dataset, xr.DataArray)):
            # Create empty xarray Dataset with same time coordinate
            empty_targets = xr.Dataset(coords={"time": features.coords["time"]})
        elif isinstance(features, (mikeio.Dataset, mikeio.DataArray)):
            # Create empty xarray Dataset with same time index as mikeio data
            empty_targets = xr.Dataset(coords={"time": features.time})
        elif isinstance(features, TimeseriesData):
            # Create empty xarray Dataset with same time coordinate as TimeseriesData
            empty_targets = xr.Dataset(coords={"time": features.time})
        else:
            raise TypeError(f"Unsupported features type: {type(features)}")

        return Timeseries(
            features=features,
            targets=empty_targets,
            item_attrs=item_attrs,
            global_attrs=global_attrs,
        )

    @staticmethod
    def empty():
        """Create an empty Timeseries.

        This factory method makes it explicit that you're creating a timeseries
        for making predictions in production, where only features are available
        and no target values exist yet.

        Parameters
        ----------
        None

        Returns
        -------
        Timeseries
            Empty Timeseries

        Returns
        -------
        Timeseries
            Empty Timeseries

        """
        from blue_ml.timeseries.timeseries import Timeseries

        return TimeseriesFactory.for_training(
            features=pd.DataFrame(), targets=pd.DataFrame()
        )

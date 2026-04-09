from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import sklearn.preprocessing as sk_scalers  # type: ignore[import-untyped]
import xarray as xr
from pandas import Timedelta
from pandas.tseries.offsets import DateOffset
from scipy.fft import fft, fftfreq  # type: ignore[import-untyped]
from scipy.signal import find_peaks  # type: ignore[import-untyped]

from blue_ml._utils import warn
from blue_ml.analysis.utils import crosscorrelation
from blue_ml.machinelearning.architectures.base_class import ONNX_FLOAT_FORMAT
from blue_ml.timeseries.timeseries import ItemClass, Timeseries
from blue_ml.timeseries.timeseries_item import ItemAlias
from blue_ml.timeseries.transforms._base_class import (
    GenericVariableTransformer,
    LimitedVariableTransformer,
    _SampleTransformer,
)


class AppliedFunction(GenericVariableTransformer):
    """Apply a function to each data variable in this dataset.

    Parameters
    ----------
    func : Callable
        Function to apply to the data
    items : str | Iterable[str], optional
        Item or items on which to apply the function, by default None. If None, the function is applied to all items
    attr_equals : dict of str:str or str:list of str
        All items with attribute attr_equals.key == attr_equals.value will be converted.
        For more than one key-value pair or key-list pair, the union (i.e. all) of the matching items will be converted.
    """

    def __init__(
        self,
        func: Callable,
        new_attrs: Optional[dict] = None,
        **kwargs,
    ):
        self._load_items_and_attr_equals(kwargs)
        self._new_attrs = new_attrs
        self.func = func

    def transform(self, x: Timeseries, **params) -> Timeseries:
        x_out = x.copy()
        items = self._get_items_to_transform(x)
        for item in items:
            x_out[item] = self.func(x_out[item].values)

        # Set/Update attrs
        final_attrs = x.attrs.copy() if x.attrs is not None else {}
        if self._new_attrs is not None:
            for name_i in items:
                final_attrs[name_i] = final_attrs.get(name_i, {}).copy()
                final_attrs[name_i].update(self._new_attrs)
        x_out.set_item_attrs(final_attrs)
        return x_out


class IncludeTemporalDifference(GenericVariableTransformer):
    """Transformer that adds the temporal differences of the passed features.

    Parameters
    ----------
    order : int, optional
        Order of differencing, by default 1
    """

    preffix = "d"

    def __init__(self, order: int = 1, **kwargs):
        self._load_items_and_attr_equals(kwargs)
        self.order = order

    def transform(self, x: Timeseries, **params) -> Timeseries:
        x_out = x.copy()

        items = self._get_items_to_transform(x)
        for item in items:
            # Convert to series for diff operation, then back to values
            series = pd.Series(x_out[item].values)
            x_out[item] = series.diff(self.order).values
            new_name = ItemAlias.from_variable_name(item).update_name(
                preffix=self.preffix
            )
            x_out.rename({item: new_name}, inplace=True)

        return x_out.sel(time=x_out.time[self.order :])


class IncludeLaggedFeatures(LimitedVariableTransformer):
    """Transformer that lags passed features.

    Parameters
    ----------
    lags : Union[int, List[int]], optional
        Range of lags to look for correlations, by default -6
    grouped_order : bool, optional
        _description_, by default False

    """

    def __init__(
        self,
        lags: Union[int, List[int]] = -6,
        grouped_order: bool = False,
        replace: bool = False,
        **kwargs,
    ):
        self._load_items_and_attr_equals(kwargs)

        if isinstance(lags, int):
            lags = [lags]
        self.lags = lags
        self.grouped_order = grouped_order
        self.replace = replace

    def transform(self, x: Timeseries, **params) -> Timeseries:
        x_out = x.copy()

        items = self._get_items_to_transform(x)
        items_ordered = x.names
        items_ordered_2d = [[i] if not self.replace else [] for i in items_ordered]
        new_features, new_targets = {}, {}
        for item in items:
            idx = items_ordered.index(item)

            for lag in self.lags:
                # Convert to series for shift operation
                series = pd.Series(x[item].values, index=x.time)
                ds_lagged = series.shift(-lag)
                name_lagged = ItemAlias.from_variable_name(item).update_name(
                    suffix=f"T{lag}"
                )
                # Load and update attrs per item
                attrs = x[item].attrs.copy()
                attrs["lag"] = lag  # type: ignore[assignment]
                new_data_array = (["time"], ds_lagged.values, attrs)
                # Add item to dataset
                if x.contains_item(item, item_class=ItemClass.FEATURE):
                    new_features[name_lagged] = new_data_array
                elif x.contains_item(item, item_class=ItemClass.TARGET):
                    new_targets[name_lagged] = new_data_array
                else:
                    raise ValueError("Item is neither feature nor target")

                items_ordered_2d[idx].append(name_lagged)

        if len(new_features) > 0:
            x_out.add_features(xr.Dataset(new_features))
        if len(new_targets) > 0:
            x_out.add_targets(xr.Dataset(new_targets))

        # Can only drop after addition of new items; otherwise empty targets
        if self.replace:
            for item in items:
                x_out = x_out.drop(item)

        if self.grouped_order and x_out is not None:
            x_out = x_out[np.hstack(items_ordered_2d)]

        # Ensure we return a Timeseries object
        assert isinstance(x_out, Timeseries), "Expected Timeseries object"
        return x_out


class ReplaceZerosWithValue(GenericVariableTransformer):
    """Replace exact zeros with a value.

    By default, zeros will be replaced by a small positive number.

    Parameters
    ----------
    value : float, optional
            small alue to replace zeros with, by default 1e-8
    """

    def __init__(self, value: float = 1e-8, **kwargs):
        self._load_items_and_attr_equals(kwargs)
        self._new_value = value

    def replace_zero_with_value(
        self, x: Union[pd.Series, np.ndarray]
    ) -> Union[pd.Series, np.ndarray]:
        x_out = x.copy()
        x_out[x_out == 0] = self._new_value
        return x_out

    def transform(self, x: Timeseries, **params) -> Timeseries:
        x_out = x.copy()
        items = self._get_items_to_transform(x)
        for item in items:
            x_out[item] = self.replace_zero_with_value(x_out[item].values)

        # Set/Update attrs
        if x.attrs is not None:
            x_out.set_item_attrs(x.attrs)
        return x_out


class DecomposeDirections(LimitedVariableTransformer):
    """Transformer to convert a direction to vector components (x and y vectors).

    Parameters
    ----------
    items : str or list of str
        Directional component to convert
    attr_equals : dict of str:str or str:list of str
        All items with attribute attr_equals.key == attr_equals.value will be converted.
        For more than one key-value pair or key-list pair, the union (i.e. all) of the matching items will be converted.
    suffix_x : str, optional
        New x item name is `original_name` + `suffix_x`, by default "_X"
    suffix_y : str, optional
        New y item name is `original_name` + `suffix_x`, by default "_Y"
    units: str, optional
        Units of the input direction, either "rad" for radians or "deg" for degrees, by default "rad"

    Raises
    ------
    ValueError
        If `items` or `attr_equals` is not provided
        If `items` and `attr_equals` are provided at the same time
    """

    def __init__(
        self, suffix_x: str = "X", suffix_y: str = "Y", units: str = "rad", **kwargs
    ):
        self._load_items_and_attr_equals(kwargs)
        # Need to have all kwargs as self.attrs, otherwise sklearn's checks will fail
        self.suffix_x = suffix_x
        self.suffix_y = suffix_y
        if units not in ["rad", "deg"]:
            raise ValueError("Units should be either 'rad' or 'deg'")
        self.units = units

    def transform(self, x: Timeseries, **params) -> Timeseries:
        x_out = x.copy()

        item_names = self._get_items_to_transform(x)
        for name in item_names:
            item_alias = ItemAlias.from_variable_name(name)
            new_item_name_x = item_alias.update_name(suffix=self.suffix_x)
            new_item_name_y = item_alias.update_name(suffix=self.suffix_y)

            values = x_out[name].values
            item_attrs = x_out[name].attrs.copy()
            item_attrs["units"] = None  # type: ignore[assignment]
            item_attrs["eumUnit"] = None  # type: ignore[assignment]
            if self.units == "rad":
                transformed_data = pd.DataFrame(
                    {
                        new_item_name_x: np.cos(values),
                        new_item_name_y: np.sin(values),
                    },
                    index=x_out.time,
                )
            elif self.units == "deg":
                transformed_data = pd.DataFrame(
                    {
                        new_item_name_x: np.cos(np.deg2rad(values)),
                        new_item_name_y: np.sin(np.deg2rad(values)),
                    },
                    index=x_out.time,
                )
            new_attrs = {
                new_item_name_x: item_attrs,
                new_item_name_y: item_attrs,
            }
            if x_out.contains_item(name, item_class=ItemClass.FEATURE):
                x_out.add_features(transformed_data, attrs=new_attrs)
            elif x_out.contains_item(name, item_class=ItemClass.TARGET):
                x_out.add_targets(transformed_data, attrs=new_attrs)

            x_out = x_out.drop(name)

        assert x_out is not None, "Transform should return a valid Timeseries"
        return x_out


class ComposeDirections(LimitedVariableTransformer):
    """Transformer to convert vector components magnitude and direction.

    Parameters
    ----------
    items : str or list of str
        Directional component to convert
    attr_equals : dict of str:str or str:list of str
        All items with attribute attr_equals.key == attr_equals.value will be converted.
        For more than one key-value pair or key-list pair, the union (i.e. all) of the matching items will be converted.
    suffix_mag : str, optional
        New x item name is `original_name` + `suffix_x`, by default "_X"
    suffix_dir : str, optional
        New y item name is `original_name` + `suffix_x`, by default "_Y"

    Raises
    ------
    ValueError
        If `items` or `attr_equals` is not provided
        If `items` and `attr_equals` are provided at the same time
    """

    def __init__(
        self,
        item_x,
        item_y,
        new_name_mag: str,
        new_name_dir: str,
        is_from=True,
        **kwargs,
    ):
        # Need to have all kwargs as self.attrs, otherwise sklearn's checks will fail
        self.item_x = item_x
        self.item_y = item_y
        self.new_name_mag = new_name_mag
        self.new_name_dir = new_name_dir
        self.is_from = is_from

    def transform(self, x: Timeseries, **params) -> Timeseries:
        x_out = x.copy()

        item_attrs = {"item_x": self.item_x, "item_y": self.item_y}

        # Only forward attrs that are the same for the item pair
        attrs_x = x_out[self.item_x].attrs
        attrs_y = x_out[self.item_y].attrs
        for k, v in attrs_x.items():
            if k in attrs_y and attrs_y[k] == v:
                item_attrs[k] = v

        # Apply covnersion
        u = x_out[self.item_x].values
        v = x_out[self.item_y].values

        spd = np.sqrt(u**2 + v**2)
        dir = np.arctan2(u, v) * 180 / np.pi
        if self.is_from:
            dir = dir + 180.0  # invert to coming from
        dir = np.mod(dir, 360)  # wrap to [0, 360]

        dct_data, new_attrs = {}, {}
        if self.new_name_mag is not None:
            dct_data[self.new_name_mag] = spd
            new_attrs[self.new_name_mag] = item_attrs
        if self.new_name_dir is not None:
            dct_data[self.new_name_dir] = dir
            new_attrs[self.new_name_dir] = item_attrs

        transformed_data = pd.DataFrame(
            dct_data,
            index=x_out.time,
        )

        # I think we can safely assume that x and y are always paired
        if x_out.contains_item(self.item_x, item_class=ItemClass.FEATURE):
            x_out.add_features(transformed_data, attrs=new_attrs)
        elif x_out.contains_item(self.item_x, item_class=ItemClass.TARGET):
            x_out.add_targets(transformed_data, attrs=new_attrs)

        x_out = x_out.drop(self.item_x)
        x_out = x_out.drop(self.item_y)

        assert x_out is not None, "Transform should return a valid Timeseries"
        return x_out


class Rename(GenericVariableTransformer):
    """Transformer to rename items in the data.

    Parameters
    ----------
    rename_dict : dict of str:str
        Mapping of input:output names
    """

    def __init__(self, rename_dict: Dict[str, str], **kwargs):
        self._load_items_and_attr_equals(kwargs)
        self.rename_dict = rename_dict

    def transform(self, x: Timeseries, **params) -> Timeseries:
        x_out = x.rename(self.rename_dict)
        assert x_out is not None, "Transform should return a valid Timeseries"
        return x_out


class AssertNames(GenericVariableTransformer):
    """Component to assert that the input and output names are the same.

    Parameters
    ----------
    items : list of str
        List of input and output names
    """

    def __init__(self, *, feature_names=None, target_names=None, **kwargs):
        self._load_items_and_attr_equals(kwargs)
        self.feature_names = feature_names
        self.target_names = target_names

    def transform(self, x, **params) -> Timeseries:
        if self.feature_names is not None:
            same_features = list(self.feature_names) == list(x.features.names)
            if not same_features:
                different_features = set(self.feature_names).symmetric_difference(
                    set(x.features.names)
                )
                raise ValueError(f"Mismatch in features: '{different_features}'")
        if self.target_names is not None:
            same_targets = list(self.target_names) == list(x.targets.names)
            if not same_targets:
                different_targets = set(self.target_names).symmetric_difference(
                    set(x.targets.names)
                )
                raise ValueError(f"Mismatch in targets: '{different_targets}'")

        return x


class Resample(_SampleTransformer):
    """Transformer to resample the time axis of the data.

    Parameters
    ----------
    resample_freq : `DateOffset`, `Timedelta` or str
        The offset string or object representing target conversion.
    method : str, optional
        Method to use for resampling, by default "mean"
        Supported methods include "mean", "sum", "max", "min", "median", "std", "first", "last"

    See Also
    --------
    :class:`pandas.DataFrame.resample`
    """

    def __init__(
        self,
        resample_freq: Union[Timedelta, DateOffset, str],
        method: Literal[
            "mean", "sum", "max", "min", "median", "std", "first", "last"
        ] = "mean",
    ):
        self.resample_freq = resample_freq
        self.method = method

    def transform(self, x: Timeseries, **params) -> Timeseries:
        x_out = x.copy()
        method = self.method
        x_out = x_out.resample(str(self.resample_freq), method=method)  # type: ignore[arg-type]

        return x_out


class FillMissing(GenericVariableTransformer):
    # TODO: Add docstring
    def __init__(self, fill: str | float = "auto", **kwargs):
        self._load_items_and_attr_equals(kwargs)
        self.fill = fill

    def transform(self, x: Timeseries, **params) -> Timeseries:
        x_out = x.fill_missing_values(fill=self.fill)

        return x_out


class TimeAsFeature(GenericVariableTransformer):
    """Creates sin/cos features based on the time of the day.

    Parameters
    ----------
    GenericFeatureTransformer : _type_
        _description_
    """

    def __init__(
        self,
        *,
        period="1h",
        time_epoch="1970-1-1 00:00:00",
        item_name_cos="time_cos_%t",
        item_name_sin="time_sin_%t",
        **kwargs,
    ):
        self._load_items_and_attr_equals(kwargs)
        self.time_epoch = time_epoch
        self.period = period
        self.item_name_cos = item_name_cos.replace("%t", str(period))
        self.item_name_sin = item_name_sin.replace("%t", str(period))

    def transform(self, x: Timeseries, **params) -> Timeseries:
        ts_out = x.copy()

        # Calculate time delta
        tdelta = x.time - pd.to_datetime(self.time_epoch)

        # Calculate sin and cos of time
        time_cos = np.cos((tdelta / pd.to_timedelta(self.period)) * (2 * np.pi))
        time_sin = np.sin((tdelta / pd.to_timedelta(self.period)) * (2 * np.pi))

        # Format as pandas (to include datetime index)
        ds_time_cos = pd.Series(time_cos, index=x.time, name=self.item_name_cos)
        ds_time_sin = pd.Series(time_sin, index=x.time, name=self.item_name_sin)

        # Finally, add to the dataset
        ts_out.add_features(
            ds_time_cos,
            names=[self.item_name_cos],
            attrs={
                self.item_name_cos: {
                    "period": self.period,
                    "time_epoch": self.time_epoch,
                }
            },
        )
        ts_out.add_features(
            ds_time_sin,
            names=[self.item_name_sin],
            attrs={
                self.item_name_sin: {
                    "period": self.period,
                    "time_epoch": self.time_epoch,
                }
            },
        )

        return ts_out


class DropNaN(GenericVariableTransformer):
    # TODO: Add docstring

    def transform(self, x: Timeseries, **params) -> Timeseries:
        x_out = x.dropna()

        return x_out


class DropVars(LimitedVariableTransformer):
    # TODO: Add docstring
    def __init__(self, **kwargs):
        self._load_items_and_attr_equals(kwargs)

    def transform(self, x: Timeseries, **params) -> Timeseries:
        x_out = x.copy()

        items = self._get_items_to_transform(x)

        for item in items:
            x_out = x_out.drop(item)

        assert x_out is not None, "Transform should return a valid Timeseries"
        return x_out


class _SkScalerWrapper(GenericVariableTransformer):
    """Wrapper for sklearn scalers to work with Timeseries objects.

    While we are technically able to use SkLearn scalers directly,
    by using this wrapper we are able to fit on a Timeseries of (features, targets)
    and transform on the features only or targets only.

    Parameters
    ----------
    apply_to_all : bool, default=False
        Controls how the scaler is applied to items in the Timeseries:

        - If False (default): Each item (column) gets its own independent scaler instance.
          The scaler parameters are stored per item name, allowing flexible transformation
          even when item order changes. Best for cases where items may vary between datasets.

        - If True: A single scaler is fitted on all items together, treating them as a
          unified dataset. The scaler expects the same number and order of items during
          transform. Best for preserving relationships between items and ensuring
          consistent scaling across the entire dataset.
    """

    def __init__(
        self,
        method: type[
            Union[
                sk_scalers.StandardScaler,
                sk_scalers.MaxAbsScaler,
                sk_scalers.MinMaxScaler,
                sk_scalers.PowerTransformer,
                sk_scalers.QuantileTransformer,
                sk_scalers.RobustScaler,
            ]
        ],
        method_args=None,
        **kwargs,
    ):
        self._load_items_and_attr_equals(kwargs)
        self.method = method
        self.method_args = method_args if method_args is not None else {}
        self._apply_to_all = True
        # Type hints for the item_method attribute
        self.item_method: Union[
            Dict[
                str,
                Union[
                    sk_scalers.StandardScaler,
                    sk_scalers.MaxAbsScaler,
                    sk_scalers.MinMaxScaler,
                    sk_scalers.PowerTransformer,
                    sk_scalers.QuantileTransformer,
                    sk_scalers.RobustScaler,
                ],
            ],
            Union[
                sk_scalers.StandardScaler,
                sk_scalers.MaxAbsScaler,
                sk_scalers.MinMaxScaler,
                sk_scalers.PowerTransformer,
                sk_scalers.QuantileTransformer,
                sk_scalers.RobustScaler,
            ],
        ]

    def _set_item_method(self, x: Timeseries, items: List[str]):
        if not self._apply_to_all:
            item_method = {}
            for item in items:
                item_method[item] = self.method(**self.method_args)
                item_method[item].set_output(transform="default")
                item_method[item].fit(x[item].values.reshape(-1, 1))
            self.item_method = item_method
        else:
            item_method = self.method(**self.method_args)
            # Type narrowing: when _apply_to_all is True, item_method is a single scaler
            scaler = cast(
                sk_scalers.StandardScaler, item_method
            )  # Cast to any sklearn scaler type
            scaler.set_output(transform="default")
            # Ensure x.values is 2D array for sklearn fit
            values_2d = x.values if x.values.ndim == 2 else x.values.reshape(-1, 1)
            scaler.fit(values_2d)  # time, features
            self.item_method = scaler

    def _replace_ts_values(self, x: Timeseries, inverse: bool = False):
        if not self._apply_to_all:
            # item_method is a dict when _apply_to_all is False
            if isinstance(self.item_method, dict):
                for item, method_i in self.item_method.items():
                    if inverse:
                        f = lambda arr: method_i.inverse_transform(  # noqa: E731
                            arr.reshape(-1, 1)
                        ).ravel()
                    else:
                        f = lambda arr: method_i.transform(arr.reshape(-1, 1)).ravel()  # noqa: E731
                    x[item] = f(x[item].values)
        else:
            # item_method is a single scaler when _apply_to_all is True
            if not isinstance(self.item_method, dict):
                scaler = self.item_method
                if inverse:
                    f = lambda arr: scaler.inverse_transform(arr)  # noqa: E731
                else:
                    f = lambda arr: scaler.transform(arr)  # noqa: E731
                x_T = f(x.values)
                for i, name in enumerate(x.names):
                    x[name] = x_T[:, i]

        return x

    def get_params(self, deep=True):
        """
        Override method to get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        try:
            return super().get_params(deep=deep)
        except AttributeError:
            out = dict()
            # Just return the method args if available
            if hasattr(self, "method_args") and self.method_args:
                out.update(self.method_args)
            return out

    def fit(self, x: Any = None, y: Any = None, **params):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        x : Timeseries
            Input samples.
        y : None, optional
            Ignored, exists for compatibility with scikit-learn API.
        apply_to_all : bool, default=False
            If True, a single scaler is fitted on all items together, treating them as a
            unified dataset. The scaler expects the same number and order of items during
            transform.
            If False, each item gets its own independent scaler instance, allowing flexible
            transformation even when item order changes.
        **params : dict
            Additional fit parameters.
        """
        apply_to_all = params.get("apply_to_all", False)
        self._apply_to_all = apply_to_all
        items = self._get_items_to_transform(x)
        self._set_item_method(x, items)
        self._is_fitted = True
        return self

    def transform(self, x: Timeseries, **params):
        """
        Transform the input data.

        Parameters
        ----------
        x : Timeseries
            Input samples.

        **params : dict
            Additional transform parameters.

        Returns
        -------
        x_out : Timeseries
            Transformed samples.
        """
        x_out = self._replace_ts_values(x.copy())
        return x_out

    def fit_transform(self, X: Timeseries, y=None, apply_to_all=False, **params):  # type: ignore[override]
        """
        Fit to data, then transform it.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X : Timeseries
            Input samples.

        y : None, optional
            Target values (ignored).

        apply_to_all : bool, default=False
            If True, a single scaler is fitted on all items together, treating them as a
            unified dataset. The scaler expects the same number and order of items during
            transform.
            If False, each item gets its own independent scaler instance, allowing flexible
            transformation even when item order changes.

        **params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : Timeseries
            Transformed Timeseries.
        """
        self.fit(X, apply_to_all=apply_to_all)
        return self.transform(X)

    def inverse_transform(self, x: Timeseries):
        x_out = self._replace_ts_values(x.copy(), inverse=True)
        return x_out


class StandardScaler(_SkScalerWrapper):
    """Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample `x` is calculated as:

        z = (x - u) / s

    where `u` is the mean of the training samples or zero if `with_mean=False`,
    and `s` is the standard deviation of the training samples or one if
    `with_std=False`.

    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the samples in the training set. Mean and
    standard deviation are then stored to be used on later data using
    :meth:`transform`.

    Standardization of a dataset is a common requirement for many
    machine learning estimators: they might behave badly if the
    individual features do not more or less look like standard normally
    distributed data (e.g. Gaussian with 0 mean and unit variance).

    For instance many elements used in the objective function of
    a learning algorithm (such as the RBF kernel of Support Vector
    Machines or the L1 and L2 regularizers of linear models) assume that
    all features are centered around 0 and have variance in the same
    order. If a feature has a variance that is orders of magnitude larger
    than others, it might dominate the objective function and make the
    estimator unable to learn from other features correctly as expected.

    `StandardScaler` is sensitive to outliers, and the features may scale
    differently from each other in the presence of outliers.

    """

    def __init__(
        self, *, copy: bool = True, with_mean: bool = True, with_std: bool = True
    ) -> None:
        super().__init__(
            method=sk_scalers.StandardScaler,
            method_args={"copy": copy, "with_mean": with_mean, "with_std": with_std},
        )
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std


class MaxAbsScaler(_SkScalerWrapper):
    """Scale each feature by its maximum absolute value.

    This estimator scales and translates each feature individually such
    that the maximal absolute value of each feature in the
    training set will be 1.0. It does not shift/center the data, and
    thus does not destroy any sparsity.

    This scaler can also be applied to sparse CSR or CSC matrices.

    `MaxAbsScaler` doesn't reduce the effect of outliers; it only linearly
    scales them down.

    """

    def __init__(self, *, copy=True) -> None:
        super().__init__(method=sk_scalers.MaxAbsScaler, method_args={"copy": copy})


class MinMaxScaler(_SkScalerWrapper):
    """Transform features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between
    zero and one.

    The transformation is given by::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where min, max = feature_range.

    This transformation is often used as an alternative to zero mean,
    unit variance scaling.

    `MinMaxScaler` doesn't reduce the effect of outliers, but it linearly
    scales them down into a fixed range, where the largest occurring data point
    corresponds to the maximum value and the smallest one corresponds to the
    minimum value.
    """

    def __init__(
        self,
        feature_range=(0, 1),
        *,
        copy=True,
        clip=False,
    ) -> None:
        super().__init__(
            method=sk_scalers.MinMaxScaler,
            method_args={"feature_range": feature_range, "copy": copy, "clip": clip},
        )


class PowerTransformer(_SkScalerWrapper):
    """Apply a power transform featurewise to make data more Gaussian-like.

    Power transforms are a family of parametric, monotonic transformations
    that are applied to make data more Gaussian-like. This is useful for
    modeling issues related to heteroscedasticity (non-constant variance),
    or other situations where normality is desired.

    Currently, PowerTransformer supports the Box-Cox transform and the
    Yeo-Johnson transform. The optimal parameter for stabilizing variance and
    minimizing skewness is estimated through maximum likelihood.

    Box-Cox requires input data to be strictly positive, while Yeo-Johnson
    supports both positive or negative data.

    By default, zero-mean, unit-variance normalization is applied to the
    transformed data.
    """

    def __init__(
        self,
        method="yeo-johnson",
        *,
        standardize=True,
        copy=True,
    ) -> None:
        super().__init__(
            method=sk_scalers.PowerTransformer,
            method_args={"method": method, "copy": copy, "standardize": standardize},
        )


class QuantileTransformer(_SkScalerWrapper):
    """Transform features using quantiles information.

    This method transforms the features to follow a uniform or a normal
    distribution. Therefore, for a given feature, this transformation tends
    to spread out the most frequent values. It also reduces the impact of
    (marginal) outliers: this is therefore a robust preprocessing scheme.

    The transformation is applied on each feature independently. First an
    estimate of the cumulative distribution function of a feature is
    used to map the original values to a uniform distribution. The obtained
    values are then mapped to the desired output distribution using the
    associated quantile function. Features values of new/unseen data that fall
    below or above the fitted range will be mapped to the bounds of the output
    distribution. Note that this transform is non-linear. It may distort linear
    correlations between variables measured at the same scale but renders
    variables measured at different scales more directly comparable.
    """

    def __init__(
        self,
        *,
        n_quantiles=1000,
        output_distribution="uniform",
        ignore_implicit_zeros=False,
        subsample=10000,
        random_state=None,
        copy=True,
    ) -> None:
        super().__init__(
            method=sk_scalers.QuantileTransformer,
            method_args={
                "n_quantiles": n_quantiles,
                "output_distribution": output_distribution,
                "ignore_implicit_zeros": ignore_implicit_zeros,
                "subsample": subsample,
                "random_state": random_state,
                "copy": copy,
            },
        )


class RobustScaler(_SkScalerWrapper):
    """Scale features using statistics that are robust to outliers.

    This Scaler removes the median and scales the data according to
    the quantile range (defaults to IQR: Interquartile Range).
    The IQR is the range between the 1st quartile (25th quantile)
    and the 3rd quartile (75th quantile).

    Centering and scaling happen independently on each feature by
    computing the relevant statistics on the samples in the training
    set. Median and interquartile range are then stored to be used on
    later data using the :meth:`transform` method.

    Standardization of a dataset is a common preprocessing for many machine
    learning estimators. Typically this is done by removing the mean and
    scaling to unit variance. However, outliers can often influence the sample
    mean / variance in a negative way. In such cases, using the median and the
    interquartile range often give better results.
    """

    def __init__(
        self,
        *,
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        copy=True,
        unit_variance=False,
    ) -> None:
        super().__init__(
            method=sk_scalers.RobustScaler,
            method_args={
                "with_centering": with_centering,
                "with_scaling": with_scaling,
                "quantile_range": quantile_range,
                "unit_variance": unit_variance,
                "copy": copy,
            },
        )


class CastTransformer(GenericVariableTransformer):
    """Cast features into a specific type.

    Adapted from skl2onnx.sklapi.CastTransformer.
    This should be used to minimize the conversion
    of a pipeline using float32 instead of double.

    Parameters
    ----------
    dtype : numpy type,
        output are cast into that type
    """

    def __init__(self, *, dtype=ONNX_FLOAT_FORMAT):
        self.dtype = dtype

    def _cast(self, ts: Timeseries, name: str):
        try:
            a2 = ts.as_xarray().astype(ONNX_FLOAT_FORMAT)
        except ValueError as e:
            # Use generic type description since Timeseries doesn't have dtype
            raise ValueError(
                "Unable to cast {} from original type into {}.".format(name, self.dtype)
            ) from e
        return a2

    def transform(self, x, **params) -> Timeseries:
        """Casts array X."""
        X_out = self._cast(x, "X")
        return Timeseries(X_out, features=x.features.names, targets=x.targets.names)


class DropEmptyAttrs(GenericVariableTransformer):
    """Transformer clean attributes which may cause issues in serialization of the data.

    E.g. None and Bool types.

    Parameters
    ----------
    replacevalue : Any, optional
        New attr value. If None, will be dropped

    """

    def __init__(self, *, replacevalue=None):
        self.replacevalue = replacevalue

    # TODO: remove _ds calls
    def transform(self, x, y=None, **params):
        """Casts array X."""
        raise NotImplementedError("Need to remove '_ds' calls from this transformation")
        # ts_out = X.copy()
        # for data_var in ts_out._ds.data_vars:
        #     attrs = ts_out._ds[data_var].attrs

        #     for k in list(attrs.keys()):
        #         if attrs[k] is None:
        #             if self.replacevalue:
        #                 ts_out._ds[data_var].attrs[k] = self.replacevalue
        #             else:
        #                 del attrs[k]
        # for k in list(ts_out._ds.attrs.keys()):
        #     if ts_out._ds.attrs[k] is None:
        #         if self.replacevalue:
        #             ts_out._ds.attrs[k] = self.replacevalue
        #         else:
        #             del ts_out._ds.attrs[k]

        # return ts_out

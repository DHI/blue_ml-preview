"""Base class definitions for model architectures."""

import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from blue_ml.io._io import ZipIO
from blue_ml.timeseries import Timeseries, TimeseriesFactory
from blue_ml.timeseries.timeseries_data import TimeseriesData


FLOAT_FORMAT = np.float32


class MLUtilsMixin:
    """Mixin class providing train/test splitting utilities for ML models."""

    def _initialize_train_test_slices(self, train_pct: float = 0.75):
        """
        Initialize train/test split parameters.

        Parameters
        ----------
        train_pct : float, optional
            Percentage of data to use for training (0, 1]. Default is 0.75.

        Raises
        ------
        ValueError
            If train_pct is not in range (0, 1].
        """
        if 0 < train_pct <= 1:
            self._train_pct = train_pct
            self._train_ts: Optional[Timeseries] = None
            self._test_ts: Optional[Timeseries] = None
        else:
            raise ValueError("'train_pct' should be a float (0, 1]")

    @property
    def is_fit(self) -> bool:
        """
        Check if the model has been fitted.

        Returns
        -------
        bool
            True if model has been fitted, False otherwise.
        """
        return self._train_ts is not None

    @property
    def train_pct(self) -> float:
        return self._train_pct

    @train_pct.setter
    def train_pct(self, other: float):
        self._train_pct = other

    def train_test_split(
        self,
        ts: Timeseries,
        split_train_test: bool = False,
    ):
        """Split data into subsets, e.g. train, val and test."""
        if not split_train_test:
            self._train_ts = ts
            self._test_ts = TimeseriesFactory.empty()
            return

        i_cut = int(self.train_pct * ts.n_time)
        train_slice = ts.sel(time=ts.time[:i_cut])
        test_slice = ts.sel(time=ts.time[i_cut:])

        self._train_ts = train_slice
        self._test_ts = test_slice

    @property
    def timeseries_train(self) -> Timeseries | None:
        return self._train_ts

    @timeseries_train.setter
    def timeseries_train(self, ts: Timeseries | None):
        self._train_ts = ts

    @property
    def timeseries_test(self) -> Timeseries | None:
        return self._test_ts

    @timeseries_test.setter
    def timeseries_test(self, ts: Timeseries | None):
        self._test_ts = ts


class BlueMLModel(ABC, MLUtilsMixin):
    """
    Abstract base class for Blue ML machine learning models.

    This class defines the interface for all machine learning models in the blue_ml
    package, providing common functionality for fitting, prediction, and serialization.

    Attributes
    ----------
    model : Any
        The underlying machine learning model.
    _feature_names : list of str
        Names of feature variables.
    _target_names : list of str
        Names of target variables.
    _feature_dtypes : dict
        Data types of features.
    _target_dtypes : dict
        Data types of targets.
    _name : str
        Model name.
    _use_rfecv : bool
        Whether to use recursive feature elimination with cross-validation.
    """

    # Define expected attributes that subclasses should implement
    model: Any  # This will be set by subclasses
    _feature_names: List[str]
    _target_names: List[str]
    _feature_dtypes: Dict[Any, Any]  # Keys can be Hashable, values are numpy dtypes
    _target_dtypes: Dict[Any, Any]  # Keys can be Hashable, values are numpy dtypes
    _name: str
    _use_rfecv: bool  # Used by some regression models

    @abstractmethod
    def __init__(self):
        pass

    def copy(self):
        """
        Create a deep copy of the model.

        Returns
        -------
        BlueMLModel
            Deep copy of the model.
        """
        return deepcopy(self)

    def fit(self, ts: Timeseries, auto_split=False):
        """
        Fit the model to training data.

        Parameters
        ----------
        ts : Timeseries
            Timeseries object containing features and targets.
        auto_split : bool, optional
            Whether to automatically split the data into training and testing sets. Default is False.

        Raises
        ------
        ValueError
            If training data is None after train_test_split.
        """
        self.train_test_split(
            ts, split_train_test=auto_split
        )  # Default to no split for now;

        # Ensure _train_ts is not None after splitting
        if self._train_ts is None:
            raise ValueError("Training data is None after train_test_split")

        self._target_names = self._train_ts.targets.names
        self._feature_names = self._train_ts.features.names
        # Onnx requires dtypes
        self._feature_dtypes = {
            d: v for d, v in self._train_ts.features.to_dataset().dtypes.items()
        }
        self._target_dtypes = {
            d: v for d, v in self._train_ts.targets.to_dataset().dtypes.items()
        }

        X_ = self._train_ts.features.values
        y_ = self._train_ts.targets.values
        if y_.shape[1] == 1:
            y_ = y_.ravel()

        self.model.fit(X=X_, y=y_)  # type: ignore[attr-defined]

    def _set_ts_for_prediction(
        self, ts: None | Timeseries, use_test: bool
    ) -> TimeseriesData:
        passed_ts = ts is not None
        if (not passed_ts) and (not use_test):
            raise ValueError("No series was passed.")
        if use_test:
            if self._test_ts is not None:
                if passed_ts:
                    warnings.warn(
                        "A series was passed with 'use_test = True' so the series will be ignored and the test series will be used instead."
                    )
                return_ts = self._test_ts
            else:
                raise ValueError("'_test_ts' is None")
        else:
            if ts is None:
                raise ValueError("No series was passed.")
            return_ts = ts

        if isinstance(return_ts, Timeseries):
            return return_ts.features
        else:
            # This shouldn't happen in practice given the logic above,
            # but we need to handle this case for the type checker
            raise ValueError("Expected Timeseries object")

    def predict(
        self, ts: Optional[Timeseries] = None, *, use_test: bool = False
    ) -> Timeseries:
        ts_features = self._set_ts_for_prediction(ts, use_test)
        # Casting to float32 to comply with onnx requirements
        X_ = ts_features.values.astype(FLOAT_FORMAT)
        # TODO: This step should not be necessary. There is a problem
        # when sklearn calls a transform method, that could be overwriting
        # our implementation. Then, sklearn require a specific formatting
        # that X_ is not following. We should look into how: TransformerMixin and
        # _SetOutputMixin are being used.
        if self._use_rfecv:  # type: ignore[attr-defined]
            estimator_ = self.model.estimator_  # type: ignore[attr-defined]
            transform = self.model._transform  # type: ignore[attr-defined]
            pred_values = estimator_.predict(transform(X_))
        else:
            pred_values = self.model.predict(X_)  # type: ignore[attr-defined]

        pred_as_da = self._convert_numpy_to_dataarray(pred_values, ts_features.time)

        # TODO: Fix and test assigning values of predictions
        if "lead_time" in pred_as_da.coords:
            pred_as_xa = pred_as_da.to_dataset(dim="lead_time")
            pred_as_xa = pred_as_xa.rename(
                {
                    old: new
                    for old, new in zip(
                        pred_as_da["lead_time"].values, self._target_names
                    )
                }
            )

            ts = Timeseries(features=ts_features, targets=pred_as_xa)

        else:
            pred_as_da.name = self._target_names[0]
            ts = Timeseries(features=ts_features, targets=pred_as_da)

        return ts

    @staticmethod
    def _convert_numpy_to_dataarray(
        x: np.ndarray, time_index: pd.Index
    ) -> xr.DataArray:
        dim0, dim1 = "time", "lead_time"
        coords = {dim0: time_index}
        output_shape = x.shape
        is_multioutput = len(output_shape) > 1
        if is_multioutput:
            dims: Union[str, tuple[str, str]] = (dim0, dim1)
            n_output = output_shape[1]
            coords[dim1] = pd.Index(range(n_output))
        else:
            dims = dim0
        return xr.DataArray(x, dims=dims, coords=coords)

    @staticmethod
    def read(filename: str) -> "BlueMLModel":
        """Read a `BlueMLModel` from a file.

        Parameters
        ----------
        filename : str
            File to read from

        Returns
        -------
        BlueMLModel
            Generic Blue_ML machine learning model

        Raises
        ------
        ValueError
            If the object loaded from the file is not a `BlueMLModel`
        """
        with open(filename, "rb"):
            obj = ZipIO.read_zip(filename)["blue_ml_model"]
        if not isinstance(obj, (BlueMLModel)):
            raise ValueError(f"Object loaded from {filename} is not a valid Model.")

        return obj

    # Sort of duplicate for the above, but this isn't on the dictionary
    def write(self, filename: str):
        """Write the model to a file.

        Parameters
        ----------
        filename : str
            Path to the output file
        """
        ZipIO.write_zip(filename, {"blue_ml_model": self})

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, model_name: str):
        self._name = model_name

    @property
    def n_targets(self) -> int:
        return len(self._target_names)

    @property
    def n_features(self) -> int:
        return len(self._feature_names)

    @property
    def target_names(self) -> list[str]:
        return self._target_names

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    @property
    def _describe(self) -> Optional[str]:
        """Additional information to include in the model's string representation."""
        return None

    def __repr__(self) -> str:
        def _print_model_info(ts):
            if hasattr(ts, "time") and len(ts.time) > 0:
                ts_report_time = ts.report.time()
                time_start = ts.time[0].strftime("%Y-%m-%d %H:%M:%S")
                time_end = ts.time[-1].strftime("%Y-%m-%d %H:%M:%S")
                steps = ts_report_time["n"]
                return f"[{time_start} - {time_end}, {steps} steps]"
            else:
                return "*None*"

        def _print_additional_model_info(model):
            if hasattr(model, "_describe") and model._describe is not None:
                return f"  {model._describe}"
            else:
                return None

        def _generate_repr(model):
            repr_str = []

            repr_str.append(f"<BlueML - {type(model).__name__}>")
            if model._describe is not None:
                repr_str.append(_print_additional_model_info(model))
            repr_str.append(f"is_fit : {model.is_fit}")
            if model.is_fit:
                repr_str.append(
                    f"  n_features : {model.n_features}, n_targets : {model.n_targets}"
                )
                repr_str.append(
                    f"  timeseries_train : {_print_model_info(model.timeseries_train)}"
                )
                repr_str.append(
                    f"  timeseries_test : {_print_model_info(model.timeseries_test)}"
                )

            return repr_str

        return "\n".join(_generate_repr(self))

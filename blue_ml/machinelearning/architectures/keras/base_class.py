"""Base class for Keras model architectures."""

import copy
from typing import Optional

import keras  # type: ignore
import tensorflow as tf
import numpy as np
from xarray import Dataset

from blue_ml.machinelearning.architectures.base_class import BlueMLModel
from blue_ml.machinelearning.windowgenerator import WindowGenerator
from blue_ml.timeseries import Timeseries


class BlueKerasModel(BlueMLModel):
    """Base class for Keras neural network models.

    This class provides common functionality for Keras models, including train/test
    splitting, feature/target name management, and prediction handling. It serves
    as a parent class for specific Keras architectures like sequential and functional models.
    """

    def __init__(self):
        self._initialize_train_test_slices()
        self._feature_names = []
        self._target_names = []

    def _build_optimizer(self):
        """Create optimizer used during compile.

        Subclasses can override to customize optimizer construction.
        """
        return keras.optimizers.Adam()

    def _build_early_stopping(self, monitor: str, patience: int):
        """Create default early stopping callback.

        Subclasses can override to customize callback policy.
        """
        return keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience, mode="min"
        )

    def _build_window_generator(self):
        """Create window generator used by windowed training paths."""
        return WindowGenerator(
            timesteps_input=self.timesteps_width, timesteps_output=1, shift=0
        )

    def _fit_from_series(
        self,
        ts: Timeseries,
        auto_split: bool = False,
        ts_val: Optional[Timeseries] = None,
        batch_size: int = 32,
        epochs: int = 20,
        callbacks=None,
        early_stopping_patience: int = 2,
        windowed: bool = False,
    ):
        """Template method for subclasses that compile from series before fitting."""
        self.train_test_split(ts, split_train_test=auto_split)

        compile_from_series = getattr(self, "compile_from_series", None)
        if not callable(compile_from_series):
            raise ValueError("Subclass must implement compile_from_series(series)")
        compile_from_series(ts)

        return self._fit_impl(
            ts=ts,
            auto_split=auto_split,
            ts_val=ts_val,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            early_stopping_patience=early_stopping_patience,
            windowed=windowed,
        )

    @property
    def loss(self):
        """Get the loss function used by the model."""
        return self._loss

    @loss.setter
    def loss(self, value):
        """Set the loss function for the model."""
        if isinstance(value, str):
            if value == "mse":
                self._loss = keras.losses.MeanSquaredError()
            elif value == "mae":
                self._loss = keras.losses.MeanAbsoluteError()
            elif value == "huber":
                self._loss = keras.losses.Huber()
            else:
                raise ValueError(f"Unsupported loss type: {value}")
        elif isinstance(value, keras.losses.Loss):
            self._loss = value
        elif callable(value):
            self._loss = value
        else:
            raise ValueError(f"Unsupported loss type: {value}")

    def compile(self, func_model) -> None:
        """Compile the functional model with loss, optimizer, and metrics.

        Sets up the model for training with mean squared error loss and
        Adam optimizer.

        Parameters
        ----------
        func_model : keras.Model
            Functional Keras model to compile
        """
        self.model = func_model
        self.n_outputs = len(self.model.outputs)

        # Use Keras output names as the source of truth for multi-output mapping.
        self.output_names = list(self.model.output_names)
        if len(self.output_names) != self.n_outputs:
            self.output_names = [f"output_{i}" for i in range(self.n_outputs)]

        # Use a dict mapping output names to independent loss instances to avoid
        # shared-state issues with stateful or custom loss objects.
        loss = {n: copy.deepcopy(self.loss) for n in self.output_names}
        optimizer = self._build_optimizer()
        metrics = {n: keras.metrics.MeanAbsoluteError() for n in self.output_names}

        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def _format_dense_targets(self, y_values):
        if self.n_outputs == 1:
            return y_values[:, 0]
        return {self.output_names[i]: y_values[:, i] for i in range(self.n_outputs)}

    def _format_window_targets(self, x_values, y_values):
        if self.n_outputs == 1:
            return x_values, y_values[..., 0]
        y_dict = {self.output_names[i]: y_values[..., i] for i in range(self.n_outputs)}
        return x_values, y_dict

    def _prepare_dense_datasets(
        self,
        train_features,
        train_targets,
        ts_val: Optional[Timeseries],
        batch_size: int,
    ):
        x_train = train_features.values
        y_train = self._format_dense_targets(train_targets.values)
        xy_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(
            batch_size
        )

        xy_validation_data = None
        if ts_val is not None:
            val_features = ts_val.features
            val_targets = ts_val.targets
            x_val = val_features.values
            y_val = self._format_dense_targets(val_targets.values)
            xy_validation_data = tf.data.Dataset.from_tensor_slices(
                (x_val, y_val)
            ).batch(batch_size)

        return xy_train, xy_validation_data

    def _prepare_windowed_datasets(
        self,
        train_features,
        train_targets,
        ts_val: Optional[Timeseries],
        batch_size: int,
    ):
        self.window = self._build_window_generator()

        xy_train = self.window(
            train_features.values,
            train_targets.values,
            batch_size=batch_size,
        )
        xy_train = xy_train.map(self._format_window_targets)

        xy_validation_data = None
        if ts_val is not None:
            xy_val = self.window(
                ts_val.features.values,
                ts_val.targets.values,
                batch_size=batch_size,
            )
            xy_validation_data = xy_val.map(self._format_window_targets)

        return xy_train, xy_validation_data

    def write(self, filename: str):
        """Write the model to a file.

        Parameters
        ----------
        filename : str
            Path to the output file
        """
        self_copy = self.copy()
        # Remove the loss function from the copy to avoid serialization issues
        # We don't need the loss, as it will be a fitted model.
        if hasattr(self_copy, "_loss"):
            del self_copy._loss
        return BlueMLModel.write(self_copy, filename)

    # The interface for loading is inherited from BlueMLModel and does not require overriding
    def fit(
        self,
        ts: Timeseries,
        auto_split: bool = False,
        ts_val: Optional[Timeseries] = None,
        batch_size: int = 32,
        epochs: int = 20,
        callbacks=None,
        early_stopping_patience: int = 2,
    ):
        """Train the model on dense (non-windowed) input/output pairs."""
        return self._fit_impl(
            ts=ts,
            auto_split=auto_split,
            ts_val=ts_val,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            early_stopping_patience=early_stopping_patience,
            windowed=False,
        )

    def _fit_impl(
        self,
        ts: Timeseries,
        auto_split: bool = False,
        ts_val: Optional[Timeseries] = None,
        batch_size: int = 32,
        epochs: int = 20,
        callbacks=None,
        early_stopping_patience: int = 2,
        windowed: bool = False,
    ):
        """Train the sequential model on time series data.

        Automatically applies train/test splitting, creates windowed batches,
        and trains the model with optional validation data and early stopping.

        Parameters
        ----------
        ts : Timeseries
            Training time series data with features and targets
        ts_val : Timeseries, optional
            Validation time series data, by default None
        batch_size : int, optional
            Number of samples per batch, by default 32
        epochs : int, optional
            Number of training epochs, by default 20
        callbacks : list, optional
            Keras callbacks to use during training, by default None
            If None and ts_val is provided, adds EarlyStopping callback
        early_stopping_patience : int, optional
            Number of epochs with no improvement before stopping, by default 2

        Raises
        ------
        ValueError
            If training data is None after train_test_split
        """
        self.train_test_split(ts, split_train_test=auto_split)

        # Ensure _train_ts is not None after splitting
        if self._train_ts is None:
            raise ValueError("Training data is None after train_test_split")

        train_features = self._train_ts.features
        train_targets = self._train_ts.targets

        self._target_names = train_targets.names
        self._feature_names = train_features.names

        if callbacks is None:
            monitor = "loss"
            if ts_val is not None:
                monitor = "val_loss"
            early_stopping = self._build_early_stopping(
                monitor=monitor,
                patience=early_stopping_patience,
            )
            callbacks = [early_stopping]

        if windowed:
            xy_train, xy_validation_data = self._prepare_windowed_datasets(
                train_features,
                train_targets,
                ts_val,
                batch_size,
            )
        else:
            xy_train, xy_validation_data = self._prepare_dense_datasets(
                train_features,
                train_targets,
                ts_val,
                batch_size,
            )

        self.model.fit(
            xy_train,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=xy_validation_data,
        )

    def predict(
        self,
        ts: Optional[Timeseries] = None,
        *,
        use_test: bool = False,
    ):
        return self._predict_impl(ts=ts, use_test=use_test, windowed=False)

    def _predict_impl(
        self,
        ts: Optional[Timeseries] = None,
        *,
        use_test: bool = False,
        windowed: bool = False,
    ):
        # Data
        ts_feat = self._set_ts_for_prediction(ts, use_test)
        ## OPTION 1:
        if not windowed:
            x_pred = [ts_feat.values]
        ## OPTION 2:
        else:
            x_pred = self.window.make_dataset(ts_feat.values, shuffle=False)

        # Predict
        prediction = self.model.predict(
            x_pred
        )  # shape of (batch size, sequence length, targets)
        nd_pred = np.array(prediction).reshape(self.n_targets, -1).T

        if windowed:
            ## Trim features based on window size
            # as_xarray, (TimeseriesData does not have isel)
            ds_feat = ts_feat.as_xarray().isel(
                time=slice(self.window.total_window_size - 1, None)
            )
        else:
            ds_feat = ts_feat.as_xarray()

        # To Timeseries
        ## Construct "TimeseriesData" in xarray format
        ds_pred = Dataset(
            data_vars={x: ("time", v) for x, v in zip(self._target_names, nd_pred.T)},
            coords={
                "time": ds_feat.time,
            },
        )

        ts_out = Timeseries(features=ds_feat, targets=ds_pred)
        return ts_out

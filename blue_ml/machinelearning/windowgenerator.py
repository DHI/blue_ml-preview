"""Window generator for time series data."""

import numpy as np
import tensorflow as tf

from blue_ml.machinelearning.architectures.base_class import FLOAT_FORMAT


class WindowGenerator:
    """Window generator for time series data.

    This class can:
        - Handle the indexes and offsets as shown in the diagrams above.
        - Split windows of features into (features, targets) pairs.
        - Plot the content of the resulting windows.
        - Efficiently generate batches of these windows from the training, evaluation, and test data, using tf.data.Datasets.

    see https://www.tensorflow.org/tutorials/structured_data/time_series
    """

    def __init__(self, timesteps_input, timesteps_output=1, shift=0, batchsize=32):
        # Some settings
        self.batchsize = batchsize

        # Work out the window parameters.
        self.timesteps_input = timesteps_input
        self.timesteps_output = timesteps_output
        self.shift = shift

        self.total_window_size = timesteps_input + shift

        self.input_slice = slice(0, timesteps_input)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.timesteps_output
        self.targets_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.targets_slice]

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
            ]
        )

    def summary(self):
        """Print a summary of the window configuration."""
        print(
            "; ".join(
                [
                    f"Total window size: {self.total_window_size}",
                    f"Input indices: {self.input_indices}",
                    f"Label indices: {self.label_indices}",
                ]
            )
        )

    def split_window(self, features, arg1):
        """
        Split features into input and target windows.

        Given a list of consecutive inputs, this method converts them to a window
        of inputs and a window of targets.

        Parameters
        ----------
        features : tf.Tensor
            Feature tensor of shape (batch, time, features).
        arg1 : Any
            Unused parameter (kept for compatibility).

        Returns
        -------
        tuple of tf.Tensor
            (inputs, targets) pair with properly shaped tensors.
        """
        inputs = features[:, self.input_slice, : self.no_features]
        targets = features[:, self.targets_slice, self.no_features :]
        """if self.label_columns is not None:
            targets = tf.stack(
                [targets[:, :, self.label_columns_indices[name]] for name in self.label_columns_indices],
                axis=-1)"""

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.timesteps_input, None])
        targets.set_shape([None, self.timesteps_output, None])

        return inputs, targets

    def make_dataset(
        self, data, data_targets=None, shuffle=True, seed=None, batch_size=32
    ):
        """
        Create TensorFlow Dataset from time series data.

        This method takes a time series DataFrame and converts it to a tf.data.Dataset
        of (input_window, label_window) pairs using tf.keras.utils.timeseries_dataset_from_array.

        Parameters
        ----------
        data : array-like
            Input features data.
        data_targets : array-like or None, optional
            Target data. If None, no splitting is performed. Default is None.
        shuffle : bool, optional
            Whether to shuffle the dataset. Default is True.
        seed : int or None, optional
            Random seed for shuffling. Default is None.
        batch_size : int, optional
            Batch size for the dataset. Default is 32.

        Returns
        -------
        tf.data.Dataset
            TensorFlow Dataset of windowed time series data.
        """
        if data_targets is None:
            data = np.array(data, dtype=FLOAT_FORMAT)
        else:
            n_features = data.shape[1]
            data = np.concatenate([data, data_targets], axis=1, dtype=FLOAT_FORMAT)

        def split_window(features):
            """Given a list of consecutive inputs, the split_window method will convert them to a window of inputs and a window of targets."""
            inputs = features[:, self.input_slice, :n_features]
            targets = features[:, self.targets_slice, n_features:]

            # Slicing doesn't preserve static shape information, so set the shapes
            # manually. This way the `tf.data.Datasets` are easier to inspect.
            inputs.set_shape([None, self.timesteps_input, None])
            targets.set_shape([None, self.timesteps_output, None])

            return inputs, targets

        # produce batches of timeseries inputs and targets.
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            seed=seed,
            shuffle=shuffle,
            batch_size=batch_size,
        )

        if data_targets is not None:
            ds = ds.map(split_window)

        return ds

    def __call__(self, data, data_targets=None, shuffle=True, seed=None, batch_size=32):
        return self.make_dataset(
            data,
            data_targets=data_targets,
            shuffle=shuffle,
            seed=seed,
            batch_size=batch_size,
        )

    @property
    def example(self):
        """Get and cache an example batch of `inputs, targets` for plotting."""
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

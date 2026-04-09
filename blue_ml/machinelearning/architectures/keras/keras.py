"""Keras neural network model implementations."""

import warnings
from abc import abstractmethod
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Self,
    Sequence,
    Tuple,
    cast,
)

from tensorflow.keras import Input, Model  # type: ignore
from tensorflow.keras.layers import LSTM, Concatenate, Dense, Layer  # type: ignore

from blue_ml.machinelearning.architectures.keras.base_class import (
    BlueKerasModel,
)
from blue_ml.timeseries.timeseries import Timeseries


class EdgeType(Enum):
    """Edge type for functional model topology.

    Attributes
    ----------
    N : int
        Independent edges, one per target
    One : int
        Shared edge across all targets
    """

    N = 1
    One = 2
    F = 3  # Final


_EDGE_ALIAS_TO_TYPE = {
    1: EdgeType.One,
    "1": EdgeType.One,
    "one": EdgeType.One,
    "n": EdgeType.N,
}


def parse_edge_type(edge: int | str | EdgeType) -> EdgeType:
    """Parse edge type from various input formats.

    Parameters
    ----------
    edge : int, str, or EdgeType
        Edge specification: 1 or "1" for One, "n" or "N" for N

    Returns
    -------
    EdgeType
        Parsed EdgeType enum value

    Raises
    ------
    ValueError
        If edge value is not valid
    """
    if isinstance(edge, EdgeType):
        return edge

    normalized = edge.lower() if isinstance(edge, str) else edge
    if normalized in _EDGE_ALIAS_TO_TYPE:
        return _EDGE_ALIAS_TO_TYPE[normalized]

    raise ValueError(
        f"Invalid edge element '{edge}', should be one of ['1', 'N'] or their aliases."
    )


def _process_kwargs(
    units: List[int] | Tuple[int] | int, **kwargs
) -> List[Dict[str, Any]]:
    """Convert kwargs to a list of dicts with the same length as `units`.

    Parameters
    ----------
    units : int, or list or tuple of int
        Number of units in each layer
    kwargs : key-pair or key-list of pairs
        Keyword arguments to be passed to each layer
        If a single value is passed, it is used for all layers

    Returns
    -------
    List of dict
        List of sequential kwargs to be passed to each layer
    """
    if not isinstance(units, list | tuple):
        units = [units]
    n_units = len(units)

    kwargs_out = [{"units": unit} for unit in units]
    for name, value in kwargs.items():
        if isinstance(value, list | tuple):
            if len(value) != n_units:
                raise ValueError(
                    f"Invalid layer property ('{name}') format, should be as long as number of units: {n_units}"
                )
        else:
            value = [value] * n_units

        for i, value_i in enumerate(value):
            kwargs_out[i].update({name: value_i})

    return kwargs_out


def _functional_model(
    list_of_kwargs: Sequence[Mapping[str, Any]],
    edges: List[EdgeType],
    n_targets: int,
    n_features: int,
    layer_type: type[Layer] = Dense,
    end_model_kwargs: Optional[Mapping[str, Any]] = None,
) -> Model:
    """Create a functional Keras model with custom topology.

    Builds a functional model with configurable input/output structure based on
    edge specifications. Supports various patterns:
    - N,N,N: Completely independent parallel paths
    - 1,N,1: Shared input, diverge to independent, converge, diverge to outputs
    - 1,1,1: Single shared path throughout

    Parameters
    ----------
    list_of_kwargs : List[Dict]
        Layer configuration dictionaries for each intermediate layer
    edges : List[EdgeType]
        Edge structure for each intermediate step (not including the input)
        Length should be len(list_of_kwargs)
    n_targets : int
        Number of target outputs
    n_features : int
        Number of input features

    Returns
    -------
    Model
        Compiled Keras functional model
    """
    activation_final = (
        end_model_kwargs.get("activation", "linear") if end_model_kwargs else "linear"
    )

    def _initial_step() -> List[Input]:
        """Create shared model input tensors."""
        if layer_type == LSTM:
            return [Input(shape=(None, n_features), name="input")]
        return [Input(shape=(n_features,), name="input")]

    def _return_sequences_kwargs(is_last_step: bool) -> Dict[str, Any]:
        if layer_type == LSTM and not is_last_step:
            return {"return_sequences": True}
        return {}

    def _intermediate_step(
        previous_edges: List[Any],
        edge: EdgeType,
        layer_args: Mapping[str, Any],
        step_no: int,
        is_last_step: bool,
    ) -> List[Any]:
        """Create one intermediate topology step."""
        basename = f"step{step_no}_layer"
        return_sequences = _return_sequences_kwargs(is_last_step)

        if edge == EdgeType.N:
            next_edges = []
            for i in range(n_targets):
                edge_source = (
                    previous_edges[i] if len(previous_edges) > 1 else previous_edges[0]
                )
                name = f"{basename}{i}"
                next_edges.append(
                    layer_type(name=name, **return_sequences, **layer_args)(edge_source)
                )
            return next_edges

        if len(previous_edges) > 1:
            merged = Concatenate(axis=-1, name=f"step{step_no}_concat")(
                [previous_edges[j] for j in range(n_targets)]
            )
            return [layer_type(name=basename, **return_sequences, **layer_args)(merged)]

        return [
            layer_type(name=basename, **return_sequences, **layer_args)(
                previous_edges[0]
            )
        ]

    def _final_step(edge_tensors: List[Any]) -> List[Any]:
        """Create final output layers with N outputs of shape (None, 1)."""
        final_edges = []
        for i in range(n_targets):
            source_tensor = (
                edge_tensors[i] if len(edge_tensors) > 1 else edge_tensors[0]
            )
            final_edges.append(
                Dense(
                    1,
                    name=f"output_{i}",
                    activation=activation_final,
                )(source_tensor),
            )
        return final_edges

    # 1. Inputs is not an actual Layer but a list of inputs
    input_edges = _initial_step()

    # 2. Here we define the layers
    output_edges = input_edges
    for step, layer_args in enumerate(list_of_kwargs):
        is_last_step = step == len(list_of_kwargs) - 1
        output_edges = _intermediate_step(
            output_edges,
            parse_edge_type(edges[step]),
            layer_args,
            step,
            is_last_step=is_last_step,
        )

    # 3. Create the final set of layers
    output_edges = _final_step(output_edges)

    # Create the Functional Model
    return Model(inputs=input_edges, outputs=output_edges)


def _compile_from_series(
    model: BlueKerasModel,
    series: Timeseries,
    *,
    list_of_kwargs: List[Dict[str, Any]],
    tree: List[str],
    layer_type: type[Layer],
    end_model_kwargs: Optional[Mapping[str, Any]] = None,
) -> None:
    """Build a functional model from series metadata and compile it."""
    model._target_names = series.targets.names
    model._feature_names = series.features.names

    func_model = _functional_model(
        list_of_kwargs=list_of_kwargs,
        edges=cast(List[EdgeType], [parse_edge_type(edge) for edge in tree]),
        n_targets=model.n_targets,
        n_features=model.n_features,
        layer_type=layer_type,
        end_model_kwargs=end_model_kwargs,
    )

    model.compile(func_model)


class BlueKerasFunctionalTopologyModel(BlueKerasModel):
    """Shared base class for functional Keras topology models."""

    def __init__(
        self,
        units_per_layer: Optional[List[int]] = None,
        activation: str = "relu",
        loss: Literal["mse", "mae", "huber"] = "mse",
        **kwargs,
    ):
        if units_per_layer is None:
            units_per_layer = []

        self.seq_model_kwargs = _process_kwargs(
            units=units_per_layer,
            activation=activation,
            **kwargs,
        )
        self.end_model_kwargs = {"activation": "linear"}
        self.loss = loss

        super().__init__()

        edges = None
        self.tree = [edge.name for edge in self._parse_tree(edges)]

    def _parse_tree(
        self,
        edges: Optional[List[str] | List[EdgeType]],
    ) -> List[EdgeType]:
        n_layers = len(self.seq_model_kwargs)
        if edges is None:
            return [EdgeType.N] * n_layers

        if len(edges) != n_layers:
            raise ValueError(
                f"Length of edges ({len(edges)}) must be equal to length of units_per_layer ({n_layers})"
            )
        return [parse_edge_type(edge) for edge in edges]

    @abstractmethod
    def _layer_type(self) -> type[Layer]:
        """Return the Keras layer class used for hidden layers."""

    def compile_from_series(self, series: Timeseries) -> None:
        """Build and compile the functional model from time series data."""
        _compile_from_series(
            model=self,
            series=series,
            list_of_kwargs=self.seq_model_kwargs,
            tree=self.tree,
            layer_type=self._layer_type(),
            end_model_kwargs=self.end_model_kwargs,
        )


class BlueLSTM(BlueKerasFunctionalTopologyModel):
    """LSTM neural network model.

    Parameters
    ----------
    timesteps_width : int, optional
        Number of timesteps to be used as input, by default 6
    units_per_layer : int or list(int), optional
        Number of units per layer, by default [64, 64]
        If a single value is passed, a single layer is created
        Excludes the final Dense layer of 1 unit
    activation : str or list(str), optional
        Activation function for each layer, by default "tanh"
        If a single value is passed, the same activation is used for all layers
    **kwargs : dict
        Additional keyword arguments to be passed to the Dense layers
    """

    def __init__(
        self,
        timesteps_width=6,
        units_per_layer: Optional[List[int]] = None,
        activation: str = "relu",
        loss: Literal["mse", "mae", "huber"] = "mse",
        **kwargs,
    ):
        super().__init__(
            units_per_layer=units_per_layer,
            activation=activation,
            loss=loss,
            **kwargs,
        )
        self.timesteps_width = timesteps_width

    @property
    def _describe(self) -> Optional[str]:
        """Additional information to include in the model's string representation."""
        window_width = self.timesteps_width
        return f"tf.keras Functional LSTM, timesteps_width : {window_width}"

        return None

    def _layer_type(self) -> type[Layer]:
        return LSTM

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
        """Train the LSTM neural network model.

        Compiles the model from the series data structure and trains it.

        Parameters
        ----------
        ts : Timeseries
            Training time series data with features and targets
        auto_split : bool, optional
            Whether to automatically split the data into training and testing sets. Default is False.
        ts_val : Timeseries, optional
            Validation time series data, by default None
        batch_size : int, optional
            Number of samples per batch, by default 32
        epochs : int, optional
            Number of training epochs, by default 20
        callbacks : list, optional
            Keras callbacks to use during training, by default None
        early_stopping_patience : int, optional
            Number of epochs with no improvement before stopping, by default 2
        """
        self._fit_from_series(
            ts=ts,
            auto_split=auto_split,
            ts_val=ts_val,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            early_stopping_patience=early_stopping_patience,
            windowed=True,
        )

    def predict(
        self,
        ts: Optional[Timeseries] = None,
        *,
        use_test: bool = False,
    ):
        return self._predict_impl(ts=ts, use_test=use_test, windowed=True)

    def _add(self, edge_type: EdgeType, units: int, activation: str) -> Self:
        """Add a layer to the model.

        Parameters
        ----------
        edge_type : str
            Type of layer to be added, either "N" or "1"
        units : int

            Number of units in the layer
        activation : str
            Activation function for the layer

        Returns
        -------
        BlueLSTM
            New instance of BlueLSTM with the added layer
        """
        self.tree.append(edge_type.name)
        self.seq_model_kwargs.append(
            {
                "units": units,
                "activation": activation,
            }
        )

        return self

    def add_final(
        self,
        activation: Literal["relu", "sigmoid", "tanh", "softmax", "linear"] = "linear",
    ) -> Self:
        """Add a final layer to the model.

        Parameters
        ----------
        activation : Literal["relu", "sigmoid", "tanh", "softmax", "linear"]
            Activation function for the layer.

        Returns
        -------
        BlueDense
            New instance of BlueDense with the added layer
        """
        self.end_model_kwargs.update({"activation": activation})
        return self

    def add_multi(
        self,
        units: int = 64,
        activation: Literal["relu", "sigmoid", "tanh", "softmax", "linear"] = "relu",
    ) -> Self:
        """Add a multi-output layer to the model.

        Parameters
        ----------
        units : int
            Number of units in the layer
        activation : Literal["relu", "sigmoid", "tanh", "softmax", "linear"]
            Activation function for the layer.

        Returns
        -------
        BlueLSTM
            New instance of BlueLSTM with the added layer
        """
        return self._add(EdgeType.N, units, activation)

    def add_single(
        self,
        units: int = 64,
        activation: Literal["relu", "sigmoid", "tanh", "softmax", "linear"] = "relu",
    ) -> Self:
        """Add a single-output layer to the model.

        Parameters
        ----------
        units : int
            Number of units in the layer
        activation : Literal["relu", "sigmoid", "tanh", "softmax", "linear"]
            Activation function for the layer.

        Returns
        -------
        BlueLSTM
            New instance of BlueLSTM with the added layer
        """
        return self._add(EdgeType.One, units, activation)

    def add_uni(
        self,
        units: int = 64,
        activation: Literal["relu", "sigmoid", "tanh", "softmax", "linear"] = "relu",
    ) -> Self:
        """Add a single-output layer to the model.

        .. deprecated:: 0.3.0
            Use :meth:`add_single` instead.

        Parameters
        ----------
        units : int
            Number of units in the layer
        activation : Literal["relu", "sigmoid", "tanh", "softmax", "linear"]
            Activation function for the layer.

        Returns
        -------
        BlueLSTM
            New instance of BlueLSTM with the added layer
        """
        warnings.warn(
            "add_uni is deprecated since version 0.3.0 and will be removed in a future release. "
            "Use add_single instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.add_single(units, activation)

    @classmethod
    def start_multi(cls) -> "BlueLSTM":
        """Start a new BlueLSTM model with one input per target.

        .. deprecated:: 0.3.0
            Instantiate :class:`BlueLSTM` directly instead.

        Returns
        -------
        BlueLSTM
            New instance of BlueLSTM with one input per target.
        """
        warnings.warn(
            "start_multi is deprecated since version 0.3.0 and will be removed in a future release. "
            "Use BlueLSTM(timesteps_width=6, units_per_layer=[]) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(units_per_layer=[])

    @classmethod
    def start_uni(cls) -> "BlueLSTM":
        """Start a new BlueLSTM model with a single input shared by all targets.

        .. deprecated:: 0.3.0
            Instantiate :class:`BlueLSTM` directly instead.

        Returns
        -------
        BlueLSTM
            New instance of BlueLSTM with a single input shared by all targets.
        """
        warnings.warn(
            "start_uni is deprecated since version 0.3.0 and will be removed in a future release. "
            "Use BlueLSTM(timesteps_width=6, units_per_layer=[]) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(units_per_layer=[])


class BlueDense(BlueKerasFunctionalTopologyModel):
    """Dense neural network model.

    Parameters
    ----------
    units_per_layer : int or list(int), optional
        Number of units per layer, by default [64, 64]
        If a single value is passed, a single layer is created
        Excludes the final Dense layer of 1 unit
    activation : str or list(str), optional
        Activation function for each layer, by default "relu"
        If a single value is passed, the same activation is used for all layers
    loss : {"mse", "mae", "huber"}, optional
        Loss function to use during training, by default "mse"
    **kwargs : dict
        Additional keyword arguments to be passed to the Dense layers
    """

    def __init__(
        self,
        units_per_layer: Optional[List[int]] = None,
        activation: str = "relu",
        loss: Literal["mse", "mae", "huber"] = "mse",
        **kwargs,
    ):
        super().__init__(
            units_per_layer=units_per_layer,
            activation=activation,
            loss=loss,
            **kwargs,
        )

    @property
    def _describe(self) -> Optional[str]:
        """Additional information to include in the model's string representation."""
        return "tf.keras Functional Dense"

    def _layer_type(self) -> type[Layer]:
        return Dense

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
        """Train the dense neural network model.

        Compiles the model from the series data structure and trains it.

        Parameters
        ----------
        ts : Timeseries
            Training time series data with features and targets
        auto_split : bool, optional
            Whether to automatically split the data into training and testing sets. Default is False.
        ts_val : Timeseries, optional
            Validation time series data, by default None
        batch_size : int, optional
            Number of samples per batch, by default 32
        epochs : int, optional
            Number of training epochs, by default 20
        callbacks : list, optional
            Keras callbacks to use during training, by default None
        early_stopping_patience : int, optional
            Number of epochs with no improvement before stopping, by default 2
        """
        self._fit_from_series(
            ts=ts,
            auto_split=auto_split,
            ts_val=ts_val,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            early_stopping_patience=early_stopping_patience,
            windowed=False,
        )

    def predict(
        self,
        ts: Optional[Timeseries] = None,
        *,
        use_test: bool = False,
    ):
        return self._predict_impl(ts=ts, use_test=use_test, windowed=False)

    def _add(self, edge_type: EdgeType, units: int, activation: str) -> Self:
        """Add a layer to the model.

        Parameters
        ----------
        edge_type : str
            Type of layer to be added, either "N" or "1"
        units : int

            Number of units in the layer
        activation : str
            Activation function for the layer

        Returns
        -------
        BlueDense
            New instance of BlueDense with the added layer
        """
        self.tree.append(edge_type.name)
        self.seq_model_kwargs.append(
            {
                "units": units,
                "activation": activation,
            }
        )

        return self

    def add_final(
        self,
        activation: Literal["relu", "sigmoid", "tanh", "softmax", "linear"] = "linear",
    ) -> Self:
        """Add a final layer to the model.

        Parameters
        ----------
        activation : Literal["relu", "sigmoid", "tanh", "softmax", "linear"]
            Activation function for the layer.

        Returns
        -------
        BlueDense
            New instance of BlueDense with the added layer
        """
        self.end_model_kwargs.update({"activation": activation})
        return self

    def add_multi(
        self,
        units: int = 64,
        activation: Literal["relu", "sigmoid", "tanh", "softmax", "linear"] = "relu",
    ) -> Self:
        """Add a multi-output layer to the model.

        Parameters
        ----------
        units : int
            Number of units in the layer
        activation : Literal["relu", "sigmoid", "tanh", "softmax", "linear"]
            Activation function for the layer.

        Returns
        -------
        BlueDense
            New instance of BlueDense with the added layer
        """
        return self._add(EdgeType.N, units, activation)

    def add_single(
        self,
        units: int = 64,
        activation: Literal["relu", "sigmoid", "tanh", "softmax", "linear"] = "relu",
    ) -> Self:
        """Add a single-output layer to the model.

        Parameters
        ----------
        units : int
            Number of units in the layer
        activation : Literal["relu", "sigmoid", "tanh", "softmax", "linear"]
            Activation function for the layer.

        Returns
        -------
        BlueDense
            New instance of BlueDense with the added layer
        """
        return self._add(EdgeType.One, units, activation)

    def add_uni(
        self,
        units: int = 64,
        activation: Literal["relu", "sigmoid", "tanh", "softmax", "linear"] = "relu",
    ) -> Self:
        """Add a single-output layer to the model.

        .. deprecated:: 0.3.0
            Use :meth:`add_single` instead.

        Parameters
        ----------
        units : int
            Number of units in the layer
        activation : Literal["relu", "sigmoid", "tanh", "softmax", "linear"]
            Activation function for the layer.

        Returns
        -------
        BlueDense
            New instance of BlueDense with the added layer
        """
        warnings.warn(
            "add_uni is deprecated since version 0.3.0 and will be removed in a future release. "
            "Use add_single instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.add_single(units, activation)

    @classmethod
    def start_multi(cls) -> "BlueDense":
        """Start a new BlueDense model with one input per target.

        .. deprecated:: 0.3.0
            Instantiate :class:`BlueDense` directly instead.

        Returns
        -------
        BlueDense
            New instance of BlueDense with one input per target.
        """
        warnings.warn(
            "start_multi is deprecated since version 0.3.0 and will be removed in a future release. "
            "Use BlueDense(units_per_layer=[]) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(units_per_layer=[])

    @classmethod
    def start_uni(cls) -> "BlueDense":
        """Start a new BlueDense model with a single input shared by all targets.

        .. deprecated:: 0.3.0
            Instantiate :class:`BlueDense` directly instead.

        Returns
        -------
        BlueDense
            New instance of BlueDense with a single input shared by all targets.
        """
        warnings.warn(
            "start_uni is deprecated since version 0.3.0 and will be removed in a future release. "
            "Use BlueDense(units_per_layer=[]) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(units_per_layer=[])

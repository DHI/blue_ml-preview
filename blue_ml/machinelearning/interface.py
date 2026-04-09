"""High-level interface for machine learning operations."""

from enum import Enum

from blue_ml.machinelearning import architectures
from blue_ml.machinelearning.architectures.base_class import BlueMLModel


class BlueArchitecture(Enum):
    """Enumeration of available machine learning architectures in Blue ML."""

    LinearModel = "BlueLinearModel"
    RidgeRegression = "BlueRidge"
    GradientBoosting = "BlueGBoost"
    KerasDense = "BlueDense"
    KerasLSTM = "BlueLSTM"


class Interface:
    """
    Interface for accessing different machine learning model architectures.

    Provides a unified way to retrieve model architectures by name.
    """

    def __init__(self):
        self._architecture_map = {e.name: e.value for e in BlueArchitecture}

    def get_model_architecture(self, model_name: str) -> BlueMLModel:
        """
        Get the model architecture class by name.

        Parameters
        ----------
        model_name : str
            Name of the model architecture (e.g., 'GlobalBaseline', 'LinearModel').

        Returns
        -------
        BlueMLModel
            The model architecture class.
        """
        blue_model_name = self._architecture_map[model_name]
        return getattr(architectures, blue_model_name)

    def print_model_options(self):
        """Print available model architecture options."""
        print(list(self._architecture_map.keys()))

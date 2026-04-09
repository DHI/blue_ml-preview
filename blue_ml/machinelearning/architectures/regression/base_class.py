"""Base class for regression models."""

from abc import abstractmethod
from typing import Optional

from blue_ml.machinelearning.architectures.base_class import BlueMLModel


class BlueRegressionModel(BlueMLModel):
    """
    Abstract base class for regression models in blue_ml.

    Provides common functionality for sklearn-based regression models.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to model initialization.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "This model is not implemented in the preview release."
        )

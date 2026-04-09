"""Time series data structures and operations."""

from blue_ml.timeseries.timeseries import Timeseries
from blue_ml.timeseries.timeseries_data import TimeseriesData
from blue_ml.timeseries.timeseries_item import TimeseriesItem

# New modular components for SOLID principles
from blue_ml.timeseries.converters import FormatConverter
from blue_ml.timeseries.io import TimeseriesReader, TimeseriesWriter
from blue_ml.timeseries.operations import (
    MissingValueHandler,
    TimeseriesModifier,
    TimeseriesResampler,
    TimeseriesSelector,
    train_val_test_split,
    train_test_split,
)
from blue_ml.timeseries.rendering import TimeseriesRenderer
from blue_ml.timeseries.attributes import AttributeManager
from blue_ml.timeseries.factories import TimeseriesFactory
from blue_ml.timeseries.ml import MLAdapter

__all__ = [
    # Core classes
    "Timeseries",
    "TimeseriesData",
    "TimeseriesItem",
    # Modular components
    "FormatConverter",
    "TimeseriesReader",
    "TimeseriesWriter",
    "TimeseriesModifier",
    "TimeseriesSelector",
    "TimeseriesResampler",
    "MissingValueHandler",
    "TimeseriesRenderer",
    "AttributeManager",
    "TimeseriesFactory",
    "MLAdapter",
    # Splitting utilities
    "train_val_test_split",
    "train_test_split",
]

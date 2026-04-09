"""Time series processing pipelines."""

from blue_ml.timeseries.pipeline._pipeline import BluePipeline

read = BluePipeline.read  # Make accessible from pipeline.read()


__all__ = [
    "read",
    "BluePipeline",
]

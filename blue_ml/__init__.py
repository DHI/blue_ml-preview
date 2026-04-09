"""Blue ML - Machine Learning for Ocean and Wave Data."""

import os
import subprocess

# Suppress TensorFlow INFO messages (keep WARNING and ERROR)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# Only disable GPU if explicitly set or if no GPU is available
# This prevents CUDA errors on CPU-only systems while allowing GPU use when available
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    try:
        # Try to detect if CUDA is available before TensorFlow loads
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=2)
        if result.returncode != 0:
            # No GPU detected, disable CUDA
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # nvidia-smi not found or timed out, assume no GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from importlib.metadata import version as get_version

from blue_ml import machinelearning
from blue_ml import datasets
from blue_ml.config import CONFIG
from blue_ml.machinelearning.windowgenerator import WindowGenerator
from blue_ml.timeseries import transforms
from blue_ml.timeseries.pipeline import BluePipeline
from blue_ml.timeseries.timeseries import Timeseries
from blue_ml.timeseries.factories import TimeseriesFactory
from blue_ml.skill.modelskill import ModelSkillAssessor
from blue_ml.modelframe import ModelFrame

__version__ = get_version(__package__)

__all__ = [
    "Timeseries",
    "TimeseriesFactory",
    "WindowGenerator",
    "BluePipeline",
    "transforms",
    "machinelearning",
    "datasets",
    "ModelSkillAssessor",
    "CONFIG",
    "ModelFrame",
]

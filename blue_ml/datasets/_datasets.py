"""Dataset loading helpers for blue_ml examples and tests.

This module provides a small, high-level API for loading bundled Timeseries
datasets from the project test-data folder.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

from blue_ml.timeseries import Timeseries
from blue_ml.timeseries import transforms
from blue_ml.timeseries.operations.splitter import train_val_test_split


from importlib.resources import files


_FILE_PREFIX = "timeseries_"
_FILE_SUFFIX = ".nc"
_DEFAULT_RELATIVE_DATA_DIR = files("blue_ml") / "datasets" / "dhi_waterbench_sw"

_HM0 = "significant_wave_height"


def _resolve_data_dir(data_dir: Optional[str | Path] = None) -> Path:
    if data_dir is not None:
        root = Path(data_dir)
    else:
        package_root = Path(__file__).resolve().parents[1]
        root = package_root / _DEFAULT_RELATIVE_DATA_DIR

    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}. "
            "Provide an explicit `data_dir` to datasets.load(...) if needed."
        )
    return root


def _to_dataset_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith(_FILE_PREFIX):
        return stem[len(_FILE_PREFIX) :]
    return stem


def _split_dataset(
    timeseries: Timeseries, train_size=0.7
) -> tuple[Timeseries, Timeseries]:
    train, test = train_val_test_split(timeseries, train_size=train_size, val_size=None)
    return train, test


def _basic_processing(
    timeseries: Timeseries, name: str, parameter: None | str = None
) -> Timeseries:
    """Basic processing for a Timeseries dataset."""
    # Isolate single parameter for prediction and drop rows with missing values
    if parameter:
        timeseries = timeseries.drop(
            list(set(timeseries.targets.names).symmetric_difference({parameter}))
        )
    timeseries = timeseries.dropna()
    # Basic transforms
    timeseries = transforms.DecomposeDirections(
        items=[f"{name}: Mean Wave Direction"]
    ).fit_transform(timeseries)
    return timeseries


def available(*, data_dir: Optional[str | Path] = None) -> list[str]:
    """Return available dataset names discovered in the dataset directory."""
    root = _resolve_data_dir(data_dir)
    datasets = [
        _to_dataset_name(path) for path in root.glob(f"{_FILE_PREFIX}*{_FILE_SUFFIX}")
    ]
    return sorted(datasets)


def path(name: str, *, data_dir: Optional[str | Path] = None) -> Path:
    """Return the on-disk path for a named dataset.

    The lookup is case-insensitive and uses discovered dataset names.
    """
    root = _resolve_data_dir(data_dir)
    index = {dataset.lower(): dataset for dataset in available(data_dir=root)}
    key = name.lower()
    if key not in index:
        raise ValueError(
            f"Unknown dataset '{name}'. Available datasets: {', '.join(sorted(index.values()))}"
        )

    canonical_name = index[key]
    return root / f"{_FILE_PREFIX}{canonical_name}{_FILE_SUFFIX}"


def load(
    name: str,
    *,
    data_dir: Optional[str | Path] = None,
) -> Timeseries:
    """Load a named Timeseries dataset.

    Examples
    --------
    >>> from blue_ml import datasets
    >>> ts = datasets.load("Europlatform2")
    """
    dataset_path = path(name, data_dir=data_dir)
    ts = Timeseries.read(str(dataset_path))
    return ts


def load_training(
    name: str, *, data_dir: Optional[str | Path] = None, split=None
) -> Timeseries | Tuple[Timeseries, Timeseries]:
    """Load a named Timeseries dataset with basic processing for training."""
    ts = load(name, data_dir=data_dir)
    ts = _basic_processing(ts, name, parameter=_HM0)
    if split:
        ts_train, ts_test = _split_dataset(ts)
        return ts_train, ts_test
    return ts


def load_training_multivariate(
    name: str, *, data_dir: Optional[str | Path] = None, split=None
) -> Timeseries | Tuple[Timeseries, Timeseries]:
    """Load a named Timeseries dataset with basic processing for training."""
    ts = load(name, data_dir=data_dir)
    ts = _basic_processing(ts, name, parameter=None)
    if split:
        ts_train, ts_test = _split_dataset(ts)
        return ts_train, ts_test
    return ts


def __getattr__(attr_name: str) -> Callable[..., Timeseries]:
    """Dynamically expose convenience loaders like `load_Europlatform2`."""
    if not attr_name.startswith("load_"):
        raise AttributeError(f"module 'datasets' has no attribute '{attr_name}'")

    dataset_name = attr_name[len("load_") :]

    def _loader(*, data_dir: Optional[str | Path] = None) -> Timeseries:
        return load(dataset_name, data_dir=data_dir)

    _loader.__name__ = attr_name
    _loader.__doc__ = (
        f"Load the '{dataset_name}' dataset. "
        "Optionally pass `data_dir` to override the default dataset directory."
    )
    return _loader


__all__ = ["available", "path", "load"]

"""Timeseries splitting utilities for train/validation/test sets."""

from typing import Optional, Tuple, overload

from blue_ml.timeseries.timeseries import Timeseries


@overload
def train_val_test_split(
    ts: Timeseries,
    train_size: float,
    val_size: None,
    test_size: Optional[float] = None,
    shuffle: bool = False,
) -> Tuple[Timeseries, Timeseries]: ...


@overload
def train_val_test_split(
    ts: Timeseries,
    train_size: float = 0.7,
    *,
    val_size: None,
    test_size: Optional[float] = None,
    shuffle: bool = False,
) -> Tuple[Timeseries, Timeseries]: ...


@overload
def train_val_test_split(
    ts: Timeseries,
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: Optional[float] = None,
    shuffle: bool = False,
) -> Tuple[Timeseries, Timeseries, Timeseries]: ...


def train_val_test_split(
    ts: Timeseries,
    train_size: float = 0.7,
    val_size: Optional[float] = 0.2,
    test_size: Optional[float] = None,
    shuffle: bool = False,
) -> Tuple[Timeseries, Timeseries] | Tuple[Timeseries, Timeseries, Timeseries]:
    """
    Split timeseries into train, validation, and test sets.

    Parameters
    ----------
    ts : Timeseries
        The timeseries to split
    train_size : float, default=0.7
        Proportion of data for training (0.0 to 1.0)
    val_size : float or None, default=0.2
        Proportion of data for validation (0.0 to 1.0).
        If None, no validation split is created and the function returns only
        (train, test).
    test_size : float, optional, default=0.1 if val_size is set, else 0.3
        Proportion of data for testing. If None, uses remainder after train and val.
    shuffle : bool, default=False
        Whether to shuffle data before splitting (not recommended for time series and not yet implemented)

    Returns
    -------
    When val_size is not None
        ts_train : Timeseries
            Training set
        ts_val : Timeseries
            Validation set
        ts_test : Timeseries
            Test set

    When val_size is None
        ts_train : Timeseries
            Training set
        ts_test : Timeseries
            Test set

    Raises
    ------
    ValueError
        If split proportions don't sum to 1.0 or are not positive

    Examples
    --------
    >>> from blue_ml.timeseries.operations import train_val_test_split
    >>> ts_train, ts_val, ts_test = train_val_test_split(ts)
    >>> ts_train, ts_val, ts_test = train_val_test_split(ts, train_size=0.6, val_size=0.3)

    Notes
    -----
    For time series data, shuffle=False (default) preserves temporal order.
    The splits are made sequentially: train | validation | test fold
    """
    # Consolidated implementation: if val_size is None, perform train/test split
    wants_val = val_size is not None
    effective_val_size = 0.0 if val_size is None else val_size

    # Validate proportions
    if test_size is None:
        test_size = 1.0 - train_size - effective_val_size

    total = train_size + effective_val_size + test_size
    if not (0.99 <= total <= 1.01):  # Allow small floating point errors
        raise ValueError(
            f"Split proportions must sum to 1.0, got {total:.3f} "
            f"(train={train_size}, val={effective_val_size}, test={test_size})"
        )

    if train_size <= 0:
        raise ValueError("train_size must be positive")

    if not (0.99 <= total <= 1.01):  # Allow small floating point errors
        if wants_val:
            msg = (
                f"Split proportions must sum to 1.0, got {total:.3f} "
                f"(train={train_size}, val={val_size}, test={test_size})"
            )
        else:
            msg = (
                f"Split proportions must sum to 1.0, got {total:.3f} "
                f"(train={train_size}, test={test_size} (val_size=None))"
            )
        raise ValueError(msg)

    if effective_val_size < 0:
        raise ValueError("val_size must be non-negative")

    if test_size <= 0:
        raise ValueError("test_size must be positive")

    # Calculate split indices
    n_time = len(ts.time)
    train_end = int(n_time * train_size)
    val_end = int(n_time * (train_size + effective_val_size))

    if shuffle:
        # TODO: Implement shuffled splitting if needed
        raise NotImplementedError("Shuffled splitting not yet implemented")

    # Sequential split (preserves temporal order)
    ts_train = ts.isel(time=slice(0, train_end))
    ts_val = ts.isel(time=slice(train_end, val_end))
    ts_test = ts.isel(time=slice(val_end, None))

    if wants_val:
        return ts_train, ts_val, ts_test

    return ts_train, ts_test


def train_test_split(
    ts: Timeseries,
    train_size: float = 0.8,
    test_size: Optional[float] = None,
    shuffle: bool = False,
) -> Tuple[Timeseries, Timeseries]:
    """
    Split timeseries into train and test sets.

    Parameters
    ----------
    ts : Timeseries
        The timeseries to split
    train_size : float, default=0.8
        Proportion of data for training (0.0 to 1.0)
    test_size : float, optional
        Proportion of data for testing. If None, uses remainder after train.
    shuffle : bool, default=False
        Whether to shuffle data before splitting

    Returns
    -------
    ts_train : Timeseries
        Training set
    ts_test : Timeseries
        Test set

    Raises
    ------
    ValueError
        If split proportions don't sum to 1.0 or are not positive

    Examples
    --------
    >>> from blue_ml.timeseries.operations import train_test_split
    >>> ts_train, ts_test = train_test_split(ts)
    >>> ts_train, ts_test = train_test_split(ts, train_size=0.7)

    Notes
    -----
    For time series data, shuffle=False (default) preserves temporal order.
    """
    # Backward-compatible wrapper around train_val_test_split
    ts_train, ts_test = train_val_test_split(
        ts,
        train_size=train_size,
        val_size=None,
        test_size=test_size,
        shuffle=shuffle,
    )

    return ts_train, ts_test

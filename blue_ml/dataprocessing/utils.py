"""Utility functions for data processing and variable selection."""

import re
from fnmatch import fnmatch
from typing import Any, Dict, List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import xarray as xr
from darts import TimeSeries as DartsTimeSeries  # type: ignore[import-untyped]
from sklearn.decomposition import PCA  # type: ignore[import-untyped]

from blue_ml._utils import warn
from blue_ml.dataprocessing.variables import variable_dictionary


def create_column_shorthand(variable_name: str) -> str:
    """Create a shortened version of variable names by extracting content from parentheses or creating abbreviations from the full name.

    Parameters
    ----------
    variable_name : str
        Full variable name

    Returns
    -------
    str
        Shortened variable name
    """
    # First try to extract from parentheses
    parentheses_content = extract_parentheses(variable_name)
    if parentheses_content:
        return parentheses_content[0]

    # If no parentheses, create abbreviation from first letters of words
    words = variable_name.split()
    if len(words) > 1:
        return "".join(word[0].upper() for word in words if word)

    # If single word, return first 3-4 characters
    return variable_name[:4].upper()


def extract_parentheses(text):
    """
    Extract text within parentheses from a string.

    Parameters
    ----------
    text : str
        Input text containing parentheses.

    Returns
    -------
    list of str
        List of strings found within parentheses.
    """
    pattern = r"\((.*?)\)"
    # Use regex to find all text within parentheses
    return re.findall(pattern, text)


def select_variable_names(**kwargs) -> List[str]:
    """Select variables from ds based on predefined variable categories.

    Accepts kwargs {"scope", "magnitude", "source", "feature"} of type ModelScope, Magnitude or WindSource, WaveFeature. See variables.py for more info.

    Returns
    -------
    List[str]
        List of variables matching the passed criteria
    """
    return [
        varname
        for varname, vardict in variable_dictionary.items()
        if all(
            (
                vardict[arg] == value  # type: ignore[index]
                if not isinstance(value, list)
                else vardict[arg] in value  # type: ignore[index]
            )
            for arg, value in kwargs.items()
        )
    ]


def select_from_dataset(
    ds: xr.Dataset,
    scale: Literal["local", "global", "all"] = "all",
    models: Optional[Union[str, List[str]]] = None,
    tslice: Optional[slice] = None,
    **kwargs,
) -> xr.Dataset:
    """Select variables from ds based on predefined variable categories.

    Accepts kwargs {"scope", "magnitude", "source", "feature"} of type ModelScope, Magnitude or WindSource, WaveFeature. See variables.py for more info.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset compiled from dfs0 files from MOOD
    tslice : slice, optional
        Time slice to filter data, by default None

    Returns
    -------
    xr.Dataset
        Subset of the original dataset containing variables matching the passed criteria
    """

    def is_all_nan(da) -> bool:
        return da.isnull().all()

    # TODO: Check if the commented text is necessary
    # if len(kwargs) > 0:
    #     assert set(kwargs.keys()).issubset({"scope", "magnitude", "source", "feature"})

    # variable_names = select_variable_names(**kwargs)

    # newds = ds.sel(variable=variable_names)
    # vars_to_drop = [var for var in newds.data_vars if is_all_nan(newds[var])]

    # newds = newds.drop_vars(vars_to_drop)

    newds = ds.copy()

    if scale == "local":
        global_vars = [var for var in newds.keys() if not newds[var].attrs["is_local"]]
        newds = newds.drop_vars(global_vars).copy()
    elif scale == "global":
        local_vars = [var for var in newds.keys() if newds[var].attrs["is_local"]]
        newds = newds.drop_vars(local_vars).copy()

    if models:
        if isinstance(models, str):
            models = [models]
        newds = newds[models].copy()

    if tslice:
        newds = newds.sel(time=tslice).copy()
    return newds.dropna(dim="variable", how="all")


def write_dataset_as_dataframe(
    ds: xr.Dataset, short_varname: bool = False
) -> pd.DataFrame:
    """
    Convert xarray Dataset to pandas DataFrame with proper indexing.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to convert.
    short_varname : bool, optional
        If True, use shortened variable names. Default is False.

    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex columns (model, variable).
    """
    df = (
        ds.to_dataframe()
        .reorder_levels(["variable", "time"])
        .reset_index(level=0)
        .pivot(columns="variable")
    )
    if short_varname:
        df.columns = pd.MultiIndex.from_tuples(
            [(col[0], create_column_shorthand(col[1])) for col in df.columns]
        )

    df.columns.names = ["model", "variable"]

    return df.dropna(axis=1, how="all")


def melt_and_create_filtering_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt DataFrame and add magnitude filtering columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with MultiIndex columns to melt.

    Returns
    -------
    pd.DataFrame
        Melted DataFrame with added magnitude column.
    """
    melted_data = (
        pd.DataFrame(df.unstack())
        .reset_index()
        .set_index("time")
        .rename(columns={0: "value"})
        .dropna()
    )
    melted_data["magnitude"] = melted_data["variable"].apply(
        lambda x: (
            mag.name
            if (mag := variable_dictionary.get(x, {}).get("magnitude")) is not None
            else "Unknown"
        )
    )

    return melted_data


def column_levels_map(df: pd.DataFrame, how: Literal["var2mod", "mod2var"] = "mod2var"):
    """
    Create a mapping of column levels in MultiIndex DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with MultiIndex columns.
    how : {'var2mod', 'mod2var'}, optional
        Direction of mapping. Default is "mod2var".

    Returns
    -------
    dict
        Dictionary mapping between column levels.

    Raises
    ------
    AssertionError
        If DataFrame does not have MultiIndex columns.
    """
    assert isinstance(df.columns, pd.MultiIndex), "Only valid for Multiindex Dataframes"
    if how == "var2mod":
        df = df.reorder_levels(order=["variable", "model"], axis=1).sort_index(axis=1)
    return {
        col: list(df[col].columns)
        for col in df.columns.get_level_values(level=0).unique()
    }


def prune_variables(
    df: pd.DataFrame,
    priority_models: Optional[Union[str, List[str]]] = [],
    models_to_drop: List[str] = [],
) -> pd.DataFrame:
    """Select the final variables to use during the modelling.

    Drops the model column row.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with multiindex (model, variable)
    priority_models : Union[str, List[str]]
        How to prioritize in case of a variable present in multiple model. Importance decreases from left to right.

    models_to_drop : List[str], optional
        List of model names that can be ruled out initially, by default []

    Returns
    -------
    pd.DataFrame
        Data with one single level of columns containing the final variables to use

    Raises
    ------
        Warning to remember that there are models chosen by default if no further user input
    """
    if not isinstance(priority_models, list):
        priority_models = [priority_models] if priority_models is not None else []

    def choose_model(variable, models):
        assert len(models) > 0, "Empty list of models!"
        if len(models) == 1:
            chosen_model = models[0]
        else:
            chosen_models = [m for m in models if m in priority_models]
            if len(chosen_models) > 0:
                return chosen_models[0]
            else:
                chosen_model = models[0]
                warn(
                    f"Multiple available models found and none is prioritized. Choosing {chosen_model} for {variable}",
                    UserWarning,
                )
        return chosen_model

    df = df.copy()
    if len(models_to_drop) > 0:
        df.drop(columns=models_to_drop, inplace=True)

    models_by_variable = column_levels_map(df, how="var2mod")
    column_pairs = []
    for variable, models in models_by_variable.items():
        chosen_model = choose_model(variable, models)
        column_pairs.append((chosen_model, variable))

    return df[column_pairs]


def filter_important_variables_by_pca(
    X: Union[pd.DataFrame, DartsTimeSeries],
    n_pca_components: Optional[int] = 8,
    loading_threshold: Optional[float] = 0.05,
    verbose: Optional[bool] = False,
):
    """
    Filter important variables using PCA loadings.

    Parameters
    ----------
    X : pd.DataFrame or DartsTimeSeries
        Input data for PCA analysis.
    n_pca_components : int or None, optional
        Number of principal components to compute. Default is 8.
    loading_threshold : float or None, optional
        Minimum absolute loading value to consider a variable important.
        Default is 0.05.
    verbose : bool or None, optional
        If True, print explained variance ratios. Default is False.

    Returns
    -------
    np.ndarray
        Array of variable names with loadings above threshold.
    """
    if isinstance(X, DartsTimeSeries):
        X = X.pd_dataframe()  # type: ignore[attr-defined]

    pca = PCA(n_components=n_pca_components)
    pca.fit(X)  # type: ignore[arg-type]

    if verbose:
        print("\n> PC explained variance ratio:")
        print(
            np.round(pca.explained_variance_ratio_, 3),
            np.round(np.sum(pca.explained_variance_ratio_), 3),
        )

    components = pd.DataFrame(pca.components_, columns=X.columns)

    # Next, I filter the components that have a loading lower than 0.05 (absolute)
    # in their principal component
    components.index = pd.Index([f"PC{i}" for i in range(1, components.shape[0] + 1)])
    components.index.name = "pc"
    components.columns.name = "variable"

    da = xr.DataArray(components.T)
    da = da.where(np.abs(da) > loading_threshold, drop=True)
    return da.coords["variable"].values


def replace_short_sequences(
    regime_list: Union[List, np.ndarray], minimum_sequence_length: int = 1
) -> np.ndarray:
    """Replace short sequences in an array of category integers.

    Given an array of integers which represent categories, it looks for sequences
    of integers that are shorter than minimum_sequence_length and fills such sequences
    with the value of the previous sequence.

    Parameters
    ----------
    regime_list : Union[List, np.ndarray]
        _description_
    minimum_sequence_length : int, optional
        _description_, by default 1

    Returns
    -------
    np.ndarray
        _description_
    """

    def evaluate_regime_label(current_regime, grouped_regimes):
        if len(current_regime) > minimum_sequence_length:
            grouped_regimes.append(current_regime)
        else:
            last_regime = grouped_regimes[-1][-1]
            grouped_regimes[-1].extend([last_regime] * len(current_regime))

        return grouped_regimes

    if isinstance(regime_list, np.ndarray):
        regime_list = list(regime_list)

    # 1) Defining first sublist
    start_first_regime = 0
    first_regime = regime_list[start_first_regime]
    start_next_regime = np.argwhere(np.array(regime_list) != first_regime)[0][0]
    first_sublist = regime_list[start_first_regime:start_next_regime]

    while len(first_sublist) <= minimum_sequence_length:
        start_first_regime = start_next_regime
        first_regime = regime_list[start_first_regime]
        start_next_regimes = np.argwhere(np.array(regime_list) != first_regime)
        start_next_regime = [
            snr for snr in start_next_regimes if snr[0] > start_first_regime
        ][0][0]
        first_sublist = regime_list[start_first_regime:start_next_regime]

    first_sublist = [first_regime] * start_first_regime + first_sublist

    regime_i = np.unique(first_sublist)
    assert len(regime_i) == 1, "More than one regime!!"

    grouped_regimes = [first_sublist]

    # 2) Fill rest of grouped regimes
    current_regime = [regime_list[start_next_regime]]
    for i, regime in enumerate(
        regime_list[start_next_regime + 1 :], start=start_next_regime + 1
    ):
        if regime == current_regime[-1]:
            current_regime.append(regime_list[i])
        else:
            grouped_regimes = evaluate_regime_label(current_regime, grouped_regimes)
            current_regime = [regime_list[i]]

    grouped_regimes = evaluate_regime_label(current_regime, grouped_regimes)

    return np.array(sum(grouped_regimes, []))  # flattening


# PLOTTING #####################################################################################


def plot_variables_by_model(df: pd.DataFrame):
    """Plot variables grouped by model from a MultiIndex DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with MultiIndex columns (model, variable)

    Raises
    ------
    ValueError
        If DataFrame does not have MultiIndex columns
    """
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("DataFrame must have MultiIndex columns")

    models, variables = df.columns.levels

    fig, axes = plt.subplots(
        len(variables), 1, sharex=True, figsize=(15, 3 * len(variables))
    )
    for i, vn in enumerate(variables):
        df.reorder_levels(["variable", "model"], axis=1)[vn].plot(ax=axes[i])
    plt.tight_layout()


def match_attrs_wildcard(attrs: Dict[str, Any], item_names: List[str]) -> Dict:
    """Match dictionary keys to list of item names with wildcard support.

    Where entries are matched, the list name is returned with the corresponding key value.

    e.g. keys{"a*":"foo", "c":"bar"} on list["a1", "a2", "b", "c"] will
    return keys{"a1":"foo", "a2":"foo", "c":"bar"}

    Parameters
    ----------
    attrs : dictionary
        Dictionary of (str, Any) pairs, where the key van be absolute or contain a wildcard "*"
    item_names : List[str]
        List of strings to compare to the keys of the attrs dictionary

    Returns
    -------
    Dict(str,Any)
        Dictionary with the matched items and their corresponding attributes
    """
    matched_attrs: Dict[str, Any] = {}
    for item_name_i, attrs_i in attrs.items():
        if "*" in item_name_i:
            items_found = [
                item for item in list(item_names) if fnmatch(item, item_name_i)
            ]
            for item_name_j in items_found:
                if item_name_j in matched_attrs.keys():
                    matched_attrs[item_name_j].update(attrs_i)
                else:
                    matched_attrs.update({item_name_j: attrs_i})
        else:
            matched_attrs.update({item_name_i: attrs_i})
    return matched_attrs

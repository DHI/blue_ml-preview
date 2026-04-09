from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sklearn.base import (  # type: ignore[import-untyped]
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
)

from blue_ml.io._io import ZipIO
from blue_ml.timeseries.timeseries import Timeseries, TimeseriesData


def read(filename: str):
    """Read a transformer from a file."""
    obj = ZipIO.read_zip(filename)
    return obj["blue_ml_transformer"]


# Since all our methods inherit from BaseEstimator, we can add a write method to it
# This also makes it simpler to export stock SkLearn transforms
def write(cls, filename: str):
    """Write a transformer to a file.

    Parameters
    ----------
    cls : BaseEstimator
        Transformer instance
    filename : str
        Output filepath
    """
    ZipIO.write_zip(filename, {"blue_ml_transformer": cls})


setattr(BaseEstimator, "write", write)


def select_items_from_attrs(
    x: Timeseries, attr_equals: Dict[str, Union[str, List]]
) -> List[str]:
    """For a given Timeseries object, select the items that have the specified attributes.

    Parameters
    ----------
    x : Timeseries
        Timeseries instance
    attr_equals : dict of str:str or str:list of str
        All items with attribute attr_equals.key == attr_equals.value will be converted.
        For more than one key-value pair or key-list pair, the union (i.e. all) of the matching items will be converted.

    Returns
    -------
    items : list of str
        List of matching items
    """
    items = []
    for name in x.names:
        for k, v_ in attr_equals.items():
            if not isinstance(v_, list):
                v_ = [v_]
            for v in v_:
                if k in x[name].attrs.keys() and v == x[name].attrs[k]:
                    items.append(name)

    if len(items) == 0:
        raise ValueError(f"No items were found with matching attributes {attr_equals}")
    return items


class _VariableTransformer(ABC, BaseEstimator, OneToOneFeatureMixin, TransformerMixin):
    """Transformer class to be used in the pipeline."""

    def fit(self, x: Any = None, y: Any = None):
        self._is_fitted = True
        return self

    @abstractmethod
    def transform(self, x, **params) -> Any:
        pass

    def _check_features_format_(self, x: Union[pd.DataFrame, Timeseries]):
        if isinstance(x, pd.DataFrame):
            self.feature_names_in_ = x.columns.tolist()
        elif isinstance(x, Timeseries) | isinstance(x, TimeseriesData):
            self.feature_names_in_ = x.names
        else:
            raise ValueError(
                "Invalid data format: only pd.Dataframe or blue_ml.Timeseries is accepted"
            )

        self.feature_names_out_ = self.feature_names_in_

    def __sklearn_is_fitted__(self):
        """Check fitted status and return a Boolean value."""
        return hasattr(self, "_is_fitted") and self._is_fitted


class _SampleTransformer(BaseEstimator, OneToOneFeatureMixin, TransformerMixin):
    def fit(self, x: Any = None, y: Any = None):
        # Custom attribute to track if the estimator is fitted
        self._is_fitted = True
        return self


class GenericVariableTransformer(_VariableTransformer):
    items: Optional[List[str]]
    attr_equals: Optional[Dict[str, Union[str, List]]]

    def _load_items_and_attr_equals(self, args: Dict):
        items_in_args = "items" in args.keys()
        attr_equals_in_args = "attr_equals" in args.keys()

        if not (items_in_args and attr_equals_in_args):
            if items_in_args:
                items = args["items"]
                if items is None:
                    raise ValueError("'items' cannot be None")
                else:
                    if not isinstance(items, list):
                        items = [items]
                    if all([isinstance(i, str) for i in items]):
                        self.items = items
                        self.attr_equals = None
                    else:
                        raise ValueError(
                            "'items' must be a string or a list of strings"
                        )
            elif attr_equals_in_args:
                attr_equals = args["attr_equals"]
                if attr_equals is None:
                    raise ValueError("'attr_equals' cannot be None")
                else:
                    self.items = None
                    self.attr_equals = attr_equals
            else:
                self.items = self.attr_equals = None
        else:
            raise ValueError(
                "Define none or only one of {items, attr_equals}. Not both."
            )

    def _get_items_to_transform(self, x: Timeseries) -> List:
        items_is_not_none = self.items is not None
        # TODO: Improve. This if statement shouldn´t be necessary
        # but there is something that happens during the pipeline
        # where items turns into an empy list
        if items_is_not_none and isinstance(self.items, list):
            items_is_not_none = len(self.items) > 0
        attr_equals_is_not_none = self.attr_equals is not None

        if items_is_not_none or attr_equals_is_not_none:
            if attr_equals_is_not_none:
                assert self.attr_equals is not None  # Type guard
                items = select_items_from_attrs(x, self.attr_equals)
            else:
                assert self.items is not None  # Type guard
                items = self.items
                items_not_found = set(items).difference(set(x.names))
                if len(items_not_found) > 0:
                    raise ValueError(
                        f"Too many items were passed. The following are not present in timeseries: {items_not_found}"
                    )
        else:
            items = x.names

        return list(items)


class LimitedVariableTransformer(_VariableTransformer):
    items: Optional[List[str]]
    attr_equals: Optional[Dict[str, Union[str, List]]]

    def _load_items_and_attr_equals(self, args: Dict):
        items_in_args = "items" in args.keys()
        attr_equals_in_args = "attr_equals" in args.keys()

        if items_in_args != attr_equals_in_args:
            if items_in_args:
                items = args["items"]
                if items is None:
                    raise ValueError("'items' cannot be None")
                else:
                    if not isinstance(items, list):
                        items = [items]
                    if all([isinstance(i, str) for i in items]):
                        self.items = items
                        self.attr_equals = None
                    else:
                        raise ValueError(
                            "'items' must be a string or a list of strings"
                        )
            else:
                attr_equals = args["attr_equals"]
                if attr_equals is None:
                    raise ValueError("'attr_equals' cannot be None")
                else:
                    self.items = None
                    self.attr_equals = attr_equals
        else:
            raise ValueError(
                "Either 'items' or 'attr_equals' need to be passed, but not both."
            )

    def _get_items_to_transform(self, x: Timeseries) -> List:
        items_is_none = self.items is None
        # TODO: Improve. This if statement shouldn´t be necessary
        # but there is something that happens during the pipeline
        # where items turns into an empy list
        if (not items_is_none) and isinstance(self.items, list):
            items_is_none = len(self.items) == 0

        if items_is_none:
            assert self.attr_equals is not None  # Type guard
            items = select_items_from_attrs(x, self.attr_equals)
        else:
            assert self.items is not None  # Type guard
            items = self.items
            items_not_found = set(items).difference(set(x.names))
            if len(items_not_found) > 0:
                raise ValueError(
                    f"Too many items were passed. The following are not present in timeseries: {items_not_found}"
                )

        return list(items)

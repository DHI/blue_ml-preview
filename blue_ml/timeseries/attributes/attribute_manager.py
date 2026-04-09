"""Attribute management for Timeseries.

This module handles all attribute-related operations for Timeseries objects,
following the Single Responsibility Principle by separating attribute management
from the core Timeseries data structure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from blue_ml.dataprocessing.utils import match_attrs_wildcard


class AttributeManager:
    """Manages item and global attributes for Timeseries.

    Responsibility: Handle all attribute storage, retrieval, and manipulation.
    """

    def __init__(
        self,
        item_attrs: Optional[Dict[str, dict]] = None,
        global_attrs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize attribute manager.

        Parameters
        ----------
        item_attrs : dict[str, dict], optional
            Attributes for each item, by default None
        global_attrs : dict[str, Any], optional
            Global attributes, by default None
        """
        self._item_attrs = item_attrs or {}
        self._global_attrs = global_attrs or {}

    @property
    def item_attrs(self) -> Dict[str, dict]:
        """Get all item attributes."""
        return self._item_attrs

    @property
    def global_attrs(self) -> Dict[str, Any]:
        """Get all global attributes."""
        return self._global_attrs

    def get_item_attrs(self, item_name: str) -> dict:
        """Get attributes for a specific item.

        Parameters
        ----------
        item_name : str
            Name of the item

        Returns
        -------
        dict
            Attributes for the item
        """
        return self._item_attrs.get(item_name, {})

    def set_item_attrs(self, item_name: str, attrs: dict) -> None:
        """Set attributes for a specific item.

        Parameters
        ----------
        item_name : str
            Name of the item
        attrs : dict
            Attributes to set
        """
        if item_name not in self._item_attrs:
            self._item_attrs[item_name] = {}
        self._item_attrs[item_name].update(attrs)

    def update_all_item_attrs(self, new_attrs: Dict[str, dict]) -> None:
        """Update attributes for multiple items.

        Parameters
        ----------
        new_attrs : dict[str, dict]
            Dictionary mapping item names to their attributes
        """
        for name, attrs in new_attrs.items():
            self.set_item_attrs(name, attrs)

    def copy_item_attrs(
        self,
        from_item: str,
        to_item: str,
        unchanged: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """Copy attributes from one item to another.

        Parameters
        ----------
        from_item : str
            Source item name
        to_item : str
            Target item name
        unchanged : str or list of str, optional
            Attribute keys that should not be changed in target item, by default ["name"]
        """
        if unchanged is None:
            unchanged = ["name"]
        elif isinstance(unchanged, str):
            unchanged = [unchanged]

        original_attributes = self.get_item_attrs(to_item)
        new_attributes = self.get_item_attrs(from_item).copy()

        for attr in unchanged:
            if attr in original_attributes:
                new_attributes[attr] = original_attributes[attr]

        self.set_item_attrs(to_item, new_attributes)

    def set_global_attrs(self, attrs: Dict[str, Any]) -> None:
        """Set global attributes.

        Parameters
        ----------
        attrs : dict
            Global attributes to set
        """
        self._global_attrs.update(attrs)

    @staticmethod
    def prepare_item_attrs(
        attrs: Dict[str, Any], other_attrs: Dict[str, Any], names: List[str]
    ) -> Dict[str, Any]:
        """Prepare item attributes by handling wildcards and per-item specifications.

        Parameters
        ----------
        attrs : dict
            Attributes with possible wildcards
        other_attrs : dict
            Additional attributes to merge
        names : list of str
            List of all item names

        Returns
        -------
        dict
            Prepared attributes per item

        Raises
        ------
        ValueError
            If attrs keys are partially but not fully in names
        """
        contains_wildcards = any("*" in k for k in attrs.keys())
        if contains_wildcards:
            attrs = match_attrs_wildcard(attrs, names)

        # Check if the passed attributes are defined by item or apply to all
        all_keys_are_names = set(attrs.keys()).issubset(set(names))
        some_keys_are_names = len(set(attrs.keys()).intersection(set(names))) > 0
        if some_keys_are_names and (not all_keys_are_names):
            raise ValueError(
                "'item_attrs' should be specified per item (including wildcards) or without any item as key."
            )

        if not all_keys_are_names:  # The attrs will be applied to all items
            attrs = {key: attrs for key in names}

        shared_keys = set(other_attrs.keys()).intersection(attrs.keys())
        if len(shared_keys) > 0:
            for var in shared_keys:
                attrs[var].update(other_attrs.pop(var))
        attrs.update(other_attrs)

        return attrs

    def remove_item_attrs(self, item_name: str) -> None:
        """Remove all attributes for a specific item.

        Parameters
        ----------
        item_name : str
            Name of the item
        """
        if item_name in self._item_attrs:
            del self._item_attrs[item_name]

    def rename_item(self, old_name: str, new_name: str) -> None:
        """Rename an item in the attribute manager.

        Parameters
        ----------
        old_name : str
            Current item name
        new_name : str
            New item name
        """
        if old_name in self._item_attrs:
            self._item_attrs[new_name] = self._item_attrs.pop(old_name)

    def copy(self) -> AttributeManager:
        """Create a deep copy of the attribute manager.

        Returns
        -------
        AttributeManager
            New attribute manager with copied attributes
        """
        import copy

        return AttributeManager(
            item_attrs=copy.deepcopy(self._item_attrs),
            global_attrs=copy.deepcopy(self._global_attrs),
        )

"""Rendering utilities for Timeseries representation.

This module handles all string and HTML rendering for Timeseries objects,
following the Single Responsibility Principle by separating display concerns
from the core Timeseries data structure.
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from xarray.core.formatting import (
    _calculate_col_width,
    attrs_repr,
    coords_repr,
    data_vars_repr,
    dim_summary_limited,
    filter_nondefault_indexes,
    indexes_repr,
    pretty_print,
    render_human_readable_nbytes,
    unindexed_dims_repr,
)
from xarray.core.formatting_html import (
    _get_indexes_dict,
    _obj_repr,
    attr_section,
    coord_section,
    datavar_section,
    dim_section,
)
from xarray.core.options import OPTIONS, _get_boolean_with_default

if TYPE_CHECKING:
    from blue_ml.timeseries.timeseries import Timeseries


class TimeseriesRenderer:
    """Handles string and HTML rendering for Timeseries.

    Responsibility: Generate text and HTML representations of Timeseries objects.
    """

    @staticmethod
    def to_string(ts: Timeseries) -> str:
        """Generate string representation of Timeseries.

        Parameters
        ----------
        ts : Timeseries
            Timeseries object to render

        Returns
        -------
        str
            String representation
        """
        ds = ts.as_xarray()

        nbytes_str = render_human_readable_nbytes(ds.nbytes)
        summary = [f"<blue_ml.Timeseries> Size: {nbytes_str}"]

        col_width = _calculate_col_width(ds.variables)
        max_rows = OPTIONS["display_max_rows"]

        dims_start = pretty_print("Dimensions:", col_width)
        dims_values = dim_summary_limited(
            ds.sizes, col_width=col_width + 1, max_rows=max_rows
        )
        summary.append(f"{dims_start}({dims_values})")

        if ds.coords:
            summary.append(
                coords_repr(ds.coords, col_width=col_width, max_rows=max_rows)
            )

        unindexed_dims_str = unindexed_dims_repr(ds.dims, ds.coords, max_rows=max_rows)
        if unindexed_dims_str:
            summary.append(unindexed_dims_str)

        summary.append(
            data_vars_repr(
                ts.features.to_dataset(),
                col_width=col_width,
                max_rows=max_rows,
                title="Features",
            )
        )

        summary.append(
            data_vars_repr(
                ts.targets.to_dataset(),
                col_width=col_width,
                max_rows=max_rows,
                title="Targets",
            )
        )

        display_default_indexes = _get_boolean_with_default(
            "display_default_indexes", False
        )
        xindexes = filter_nondefault_indexes(
            _get_indexes_dict(ds.xindexes), not display_default_indexes
        )
        if xindexes:
            summary.append(indexes_repr(xindexes, max_rows=max_rows))

        if ds.attrs:
            summary.append(attrs_repr(ds.attrs, max_rows=max_rows))

        return "\n".join(summary)

    @staticmethod
    def to_html(ts: Timeseries) -> str:
        """Generate HTML representation of Timeseries.

        Parameters
        ----------
        ts : Timeseries
            Timeseries object to render

        Returns
        -------
        str
            HTML representation
        """
        ds = ts.as_xarray()

        obj_type = "blue_ml.Timeseries"
        header_components = [f"<div class='xr-obj-type'>{escape(obj_type)}</div>"]
        sections = [
            dim_section(ds),
            coord_section(ds.coords),
            datavar_section(ts.features.to_dataset(), name="Features"),
            datavar_section(ts.targets.to_dataset(), name="Targets"),
            attr_section(ds.attrs),
        ]

        return _obj_repr(ds, header_components, sections)

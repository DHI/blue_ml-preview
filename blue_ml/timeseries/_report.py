"""
Timeseries.report.

The Timeseries.report module provides a class to generate
a report for a Timeseries object. The report contains information
about the time, missing values and warnings of the Timeseries object.

Structure

_PrettyReport (css styling and ascii printing)
    |
    +--TimeseriesReport (collection of time, missing_values and warnings sub reports)
    |

        |
        +-- TimeseriesWarningsReport (sub report for warnings, based on MissingV and Time report)
             |
             +-- TimeseriesMissingValuesReport (sub report for missing values)
             |
             +-- TimeseriesTimeReport (sub report for time)

"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from blue_ml._utils import warn

if TYPE_CHECKING:
    from blue_ml.timeseries import Timeseries


class _PrettyReport:
    _css = {
        "p": {"margin-top": "0px", "margin-bottom": "0px", "margin-left": "12px"},
        "p2": {
            "margin-top": "0px",
            "margin-bottom": "0px",
            "margin-left": "12px",
            "color": "red",
        },
        "h2": {"margin-top": "10px", "margin-bottom": "6px"},
        "h3": {"margin-top": "10px", "margin-bottom": "6px"},
        "h4": {"margin-top": "10px", "margin-bottom": "6px"},
        "h5": {"margin-top": "10px", "margin-bottom": "6px"},
    }

    def __init__(self, name: str) -> None:
        self.name = name
        return None

    def __call__(self) -> dict:
        """Override this method in subclasses."""
        return {}

    def _init_css_(self) -> List[str]:
        """Setup styling for html output.

        Returns
        -------
        List of str
            html lines for css styling
        """
        html_lines = []

        # Settings
        def define_style(css_style, **kwargs):
            lines = []
            lines.append(css_style + " {")
            for k, v in kwargs.items():
                lines.append(f"{k}: {v};")
            lines.append("}")
            return lines

        def init_styling(styles):
            html_lines.append("<html>")
            html_lines.append("<head>")
            html_lines.append("<style>")
            for style in styles:
                for s in style:
                    html_lines.append(s)
            html_lines.append("</style>")
            html_lines.append("</head>")
            html_lines.append("<body>")

        # CSS
        styles = []
        for k, v in self._css.items():
            styles.append(define_style(k, **v))
        init_styling(styles)

        return html_lines

    @staticmethod
    def print_value(
        value: str, indent: int = 0, css_style: str = "p", css: bool = False
    ) -> str:
        """Print a single value with indentation.

        Parameters
        ----------
        value : str
            value to print
        indent : int, optional
            indent spacing, by default 0
        css_style : str, optional
            css style tag, by default 'p'
        css : bool, optional
            is output plain or html formatted, by default False (plain)

        Returns
        -------
        str
            Formatted string
        """
        if css_style not in _PrettyReport._css:
            raise ValueError(
                f"CSS style '{css_style}' not defined in _PrettyReport._css"
            )

        if css:
            indent_str = "&nbsp;" * 4 * indent
            return f"<div><{css_style}>{indent_str}{value}</{css_style}></div>"
        else:
            indent_str = "    " * indent
            return f"{indent_str}{value}"

    @staticmethod
    def format_str(value: str, css_style: str = "p", css: bool = False) -> str:
        """Adds css styling tag before and after string.

        Parameters
        ----------
        value : str
            Text to format
        css_style : str, optional
            css style tag, by default 'p'
        css : bool, optional
            is output plain or html formatted, by default False (plain)

        Returns
        -------
        str
            Formatted string
        """
        if css:
            return f"<{css_style}>{value}</{css_style}>"
        else:
            return "Formatted String - idk what was supposed to be returned here lol, it just says that and was plain str type before, sorry /jannik"

    @staticmethod
    def print_key_value(
        key: str, value: str, indent: int = 0, css_style: str = "p", css: bool = False
    ) -> str:
        """Print a key-value pair with indentation.

        Applies special formatting for values with '!text!'

        Parameters
        ----------
        value, value : str
            key and value to print
        indent : int, optional
            indent spacing, by default 0
        css_style : str, optional
            css style tag, by default 'p'
        css : bool, optional
            is output plain or html formatted, by default False (plain)

        Returns
        -------
        str
            Formatted string
        """
        if css:
            print_key = _PrettyReport.format_str(key, css_style="b", css=css)

            # Special formatting for warnings in "!text!"
            if str(value).startswith("!") and str(value).endswith("!"):
                print_str = f'{print_key}: <span style="color:red">{value[1:-1]}</span>'
            else:
                print_str = f"{print_key}: {value}"

            return _PrettyReport.print_value(
                print_str, indent=indent, css_style=css_style, css=css
            )
        else:
            print_str = f"{key}: {value}"
            return _PrettyReport.print_value(
                print_str, indent=indent, css_style=css_style, css=css
            )

    @staticmethod
    def title_str(value: str, css=False) -> str:
        """Format the title of the report.

        Hardcoded to 'h2' css style

        Parameters
        ----------
        value : str
            Value to print
        css : bool, optional
            Is output plain or html formatted, by default False (plain)

        Returns
        -------
        str
            Formatted string
        """
        if css:
            return _PrettyReport.print_value(value, indent=0, css_style="h2", css=True)
        else:
            return f"<{value}>"

    def _repr_multi_(self, css: bool = False) -> str:
        """Create a _repr_ string for both html and plain text."""
        if css:
            print_str = self._init_css_()
        else:
            print_str = []

        # Set title
        print_str.append(self.title_str(self.name, css=css))

        # Get dictionsry of report
        report = self.__call__()

        def print_next(dct_, level=0):
            """Print nested dictionaries iteratively."""
            for key, value in dct_.items():
                if isinstance(value, dict):
                    css_style = f"h{level + 3}"
                    print_str.append(
                        self.print_value(
                            key, indent=level, css_style=css_style, css=css
                        )
                    )
                    print_next(value, level + 1)
                else:
                    print_str.append(
                        self.print_key_value(key, value, indent=level, css=css)
                    )

        print_next(report, 0)
        return "\n".join(print_str)

    def __repr__(self):
        return self._repr_multi_(css=False)

    def _repr_html_(self):
        return self._repr_multi_(css=True)


class TimeseriesReport(_PrettyReport):
    """Summary report of time, missing values and warnings of a Timeseries object."""

    def __init__(self, ts: Timeseries):
        self.ts = ts
        self.time = TimeseriesTimeReport(ts)
        self.missing_values = TimeseriesMissingValuesReport(ts)
        self.warnings = TimeseriesWarningsReport(self.time, self.missing_values)
        super().__init__("Timeseries.report")

    def __call__(self) -> dict:
        time = self.time()
        warnings = self.warnings()
        missing_values = self.missing_values()
        return dict(time=time, missing_values=missing_values, warnings=warnings)

    def validate(self, raise_warning=True, raise_exception=False) -> bool:
        """Validate the dataset for ML use.

        Parameters
        ----------
        warn : bool, optional
            Print warnings, by default True
        exception : bool, optional
            Raise exception if invalid, by default False

        Returns
        -------
        List[str]
            List of warnings
        """
        return self.warnings.validate(
            raise_warning=raise_warning, raise_exception=raise_exception
        )

    @property
    def is_valid(self) -> bool:
        """Check if the dataset is valid for ML use.

        see `self.report` for details

        Returns
        -------
        bool
            True if valid, False otherwise
        """
        return self.validate(raise_warning=False, raise_exception=False)


class TimeseriesWarningsReport(_PrettyReport):
    """Summary report of warnings of a Timeseries object."""

    def __init__(self, time, missing_values):
        self.TimeseriesTimeReport = time
        self.TimeseriesMissingValuesReport = missing_values
        super().__init__("Timeseries.report.warnings")

    def __call__(self) -> dict:
        warnings = {}
        if self.TimeseriesTimeReport.is_equidistant is False:
            warnings["time.is_equidistant"] = "!Time is not equidistant!"
        if self.TimeseriesTimeReport.is_sorted is False:
            warnings["time.is_sorted"] = "!Time is not sorted!"
        if (
            "features" in self.TimeseriesMissingValuesReport()
            and len(self.TimeseriesMissingValuesReport()["features"]) > 0
        ):
            warnings["missing_values.features"] = "!nans in features!"
        if (
            "targets" in self.TimeseriesMissingValuesReport()
            and len(self.TimeseriesMissingValuesReport()["targets"]) > 0
        ):
            warnings["missing_values.targets"] = "!nans in targets!"
        if len(warnings) == 0:
            warnings["noTimeseriesWarningsReport"] = "No warnings"
        return warnings

    def validate(self, raise_warning=True, raise_exception=False) -> bool:
        """Validate the dataset for ML use.

        Parameters
        ----------
        warn : bool, optional
            Print warnings, by default True
        exception : bool, optional
            Raise exception if invalid, by default False

        Returns
        -------
        List[str]
            List of warnings
        """
        warnings = self()
        if len(warnings) == 1 and "noTimeseriesWarningsReport" in warnings:
            return True
        if raise_warning:
            for k, v in warnings.items():
                warn(f"Warning: {k}: {v}")
        if raise_exception:
            raise ValueError(
                f"Dataset is invalid. The following validations failed: {', '.join(list(warnings))}"
            )
        return False

    @property
    def is_valid(self) -> bool:
        """Check if the dataset is valid for ML use.

        see `self.report` for details

        Returns
        -------
        bool
            True if valid, False otherwise
        """
        return self.validate(raise_warning=False, raise_exception=False)


class TimeseriesMissingValuesReport(_PrettyReport):
    """Summary report of warnings of a Timeseries object."""

    def __init__(self, ts):
        self.ts = ts
        super().__init__("Timeseries.report.missing_values")

    def __call__(self) -> dict:
        # Get count of missing values
        dct_missing = self.count_na()
        report_missing = {}

        # Loop through [features, targets]
        for ft, df_missing_ft in dct_missing.items():
            items_missing = {i: v for i, v in df_missing_ft.items() if v > 0}
            if len(items_missing) > 0:
                report_missing[ft] = items_missing

        if len(report_missing) == 0:
            report_missing["no_missing_values"] = {"status": "No missing values"}

        return report_missing

    def isna(self) -> dict:
        """Check for missing values.

        Returns
        -------
        dict
            Dictionary with boolean DataFrames for missing values
        """
        dct_na = {}
        for item_class in ["features", "targets"]:
            df_na = np.isnan(getattr(self.ts, item_class).to_dataframe())
            dct_na[item_class] = df_na
        return dct_na

    def count_na(self, axis=0) -> dict:
        """Count missing values for each item.

        Parameters
        ----------
        axis : int, optional
            by default 0: count for each each item,
            1: count for each time step

        Returns
        -------
        dict
            Count of missing values
        """
        dct_count = {}
        for item_class, df_na in self.isna().items():
            dct_count[item_class] = df_na.sum(axis=axis)
        return dct_count


class TimeseriesTimeReport(_PrettyReport):
    """Summary report of time of a Timeseries object."""

    def __init__(self, ts):
        self.ts = ts
        super().__init__("Timeseries.report.time")

    def __call__(self) -> dict:
        t0 = self.start
        t1 = self.end
        nt = self.n
        eq = self.is_equidistant
        is_sorted = self.is_sorted
        timestep = self.timestep
        duration_str = self.duration_str
        time = {
            "start": t0,
            "end": t1,
            "n": nt,
            "is_equidistant": eq,
            "timestep": timestep,
            "is_sorted": is_sorted,
            "duration": duration_str,
        }
        return time

    @property
    def start(self) -> pd.Timestamp:
        """Start time of the Timeseries.

        Returns
        -------
        pd.Timestamp
            Start time of the Timeseries
        """
        return self.ts.time[0]

    @property
    def end(self) -> pd.Timestamp:
        """End time of the Timeseries.

        Returns
        -------
        pd.Timestamp
            End time of the Timeseries
        """
        return self.ts.time[-1]

    @property
    def n(self) -> int:
        """Number of time steps in the Timeseries.

        Returns
        -------
        int
            Number of time steps
        """
        return len(self.ts.time)

    @property
    def is_equidistant(self) -> bool:
        """Check if the time steps are equidistant.

        Returns
        -------
        bool
            True if time steps are equidistant
        """
        if self.timestep is None:
            return False
        return True

    @property
    def is_sorted(self) -> bool:
        """Check if the time steps are sorted.

        Returns
        -------
        bool
            True if time steps are sorted
        """
        return bool(~np.any(np.diff(self.ts.time) < pd.Timedelta(0)))

    @property
    def timestep(self) -> Union[pd.Timedelta, None]:
        """Inferred timestep of the Timeseries.

        Returns
        -------
        Union[pd.Timedelta, None]
            Inferred timestep of the Timeseries
        """
        inferred_freq = self.ts.freq
        if inferred_freq is None:
            return None
        # value-less freqs, otherwise timedelta will fail
        if not np.char.isdigit(inferred_freq[0]):
            inferred_freq = "1" + inferred_freq
        return pd.to_timedelta(inferred_freq)

    @property
    def duration_str(self) -> str:
        """Duration of the Timeseries, in natural string format.

        Returns
        -------
        str
            Duration of the Timeseries
        """
        duration = self.duration
        if duration.years > 0:
            durationstr = f"{duration.years} years, {duration.months} months, {duration.days} days"
        elif duration.months > 0:
            durationstr = f"{duration.months} months, {duration.days} days, {duration.hours} hours"
        elif duration.days > 0:
            durationstr = f"{duration.days} days, {duration.hours} hours, {duration.minutes} minutes"
        elif duration.hours > 0:
            durationstr = f"{duration.hours} hours, {duration.minutes} minutes, {duration.seconds} seconds"
        else:
            durationstr = f"{duration.minutes} minutes, {duration.seconds} seconds , {duration.microseconds} microseconds"
        return durationstr

    @property
    def duration(self) -> relativedelta:
        """Duration of the Timeseries, in relativedelta format.

        Returns
        -------
        relativedelta
            Duration of the Timeseries
        """
        return relativedelta(self.ts.time[-1], self.ts.time[0])

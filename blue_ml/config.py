"""Configuration management for Blue ML."""

from typing import Any


class _CONFIG:
    """
    Configuration class for blue_ml package settings.

    This class manages global configuration options for the blue_ml package,
    including variable naming conventions and validation settings.

    Attributes
    ----------
    variable_source_separator : str
        Separator used between variable name and source in variable identifiers.
        Default is " @ ".
    allow_invalid_timeseries : bool
        Whether to allow creation of invalid timeseries objects.
        Default is True.
    supress_warnings : bool
        Whether to suppress warning messages.
        Default is False.
    """

    # Defaults
    __conf = {
        "allow_invalid_timeseries": True,
        "supress_warnings": False,
        "variable_source_separator": " @ ",
    }

    def __init__(self):
        for key in self.__conf.keys():
            setattr(self, key, self.__conf[key])

    def __getitem__(self, name: str):
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any):
        if name in self.__conf.keys():
            super().__setattr__(name, value)
        else:
            raise NameError(f"'{name}' is not a valid configuration property")

    def __repr__(self):
        return str(self.__dict__)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__conf.keys():
            super().__setattr__(name, value)
        else:
            raise NameError(f"'{name}' is not a valid configuration property")

    # Add explicit type annotations for the dynamic attributes
    # This helps type checkers understand what attributes exist
    variable_source_separator: str
    allow_invalid_timeseries: bool
    supress_warnings: bool


CONFIG = _CONFIG()

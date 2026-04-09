import warnings

from .config import CONFIG

# Global variable to control warning suppression
_suppress_warnings: bool = False


def suppress_warnings(state: bool = True):
    """
    Globally suppress all user warnings in the module.

    Parameters
    ----------
    state : bool, optional
        If True, suppress warnings. If False, enable warnings.
        Default is True.
    """
    global _suppress_warnings
    _suppress_warnings = state


def warn(message: str, category: type[Warning] | None = None, stacklevel: int = 2):
    """
    Issue a warning if warnings are not suppressed in configuration.

    Parameters
    ----------
    message : str
        The warning message to display.
    category : type[Warning] or None, optional
        The warning category. If None, uses default UserWarning.
        Default is None.
    stacklevel : int, optional
        The stack level for the warning. Default is 2.
    """
    if not CONFIG["supress_warnings"]:
        warnings.warn(message, category=category, stacklevel=stacklevel)

from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.utils._set_output import (  # type: ignore[import-untyped]
    ADAPTERS_MANAGER,
    _wrap_method_output,
    get_columns,
)

from blue_ml.io._io import ZipIO
from blue_ml.timeseries import Timeseries, TimeseriesData


class BlueAdapter:
    """Adapter for BlueML containers. Unstable, but works for proof of concept."""

    container_lib = "blueml"

    def create_container(self, X_output, X_original, columns, inplace=True):
        """Create container from `X_output` with additional metadata.

        Parameters
        ----------
        X_output : {ndarray, dataframe}
            Data to wrap.

        X_original : {ndarray, dataframe}
            Original input dataframe. This is used to extract the metadata that should
            be passed to `X_output`, e.g. pandas row index.

        columns : callable, ndarray, or None
            The column names or a callable that returns the column names. The
            callable is useful if the column names require some computation. If `None`,
            then no columns are passed to the container's constructor.

        inplace : bool, default=False
            Whether or not we intend to modify `X_output` in-place. However, it does
            not guarantee that we return the same object if the in-place operation
            is not possible.

        Returns
        -------
        wrapped_output : container_type
            `X_output` wrapped into the container type.
        """
        columns = get_columns(columns)

        if not inplace or not isinstance(X_output, Timeseries):
            if isinstance(X_output, TimeseriesData):
                return Timeseries(X_output)

            if isinstance(X_output, Timeseries):
                index = X_output.time
                columns_ = X_output.names
                attrs_ = X_output._attrs
                target_names = X_output.targets.names

            elif isinstance(X_original, Timeseries):
                index = X_original.time
                columns_ = X_original.names
                attrs_ = X_original._attrs
                target_names = X_original.targets.names

            else:
                index = None
                columns_ = None
                attrs_ = None
                target_names = None

            if isinstance(X_output, Timeseries):
                index = X_output.time
            elif isinstance(X_original, Timeseries):
                index = X_original.time
            else:
                index = None

            # We don't pass columns here because it would intend columns selection
            # instead of renaming.

            X_output = Timeseries.from_index_and_values(
                X_output,
                index=index,
                column_names=columns_,
                attrs=attrs_,
                target_names=target_names,
            )

        if columns is not None:
            Xt = self.rename_columns(X_output, columns)
            # Xt.attrs = X_output.attrs
            return Xt

        return X_output

    def is_supported_container(self, X):
        """Return True if X is a supported container.

        Parameters
        ----------
        Xs: container
            Containers to be checked.

        Returns
        -------
        is_supported_container : bool
            True if X is a supported container.
        """
        return isinstance(X, Timeseries)

    def rename_columns(self, X, columns):
        """Rename columns in `X`.

        Parameters
        ----------
        X : container
            Container which columns is updated.

        columns : ndarray of str
            Columns to update the `X`'s columns with.

        Returns
        -------
        updated_container : container
            Container with new names.
        """
        # X._names = columns
        return X

    def hstack(self, Xs):
        """Stack containers horizontally (column-wise).

        Parameters
        ----------
        Xs : list of containers
            List of containers to stack.

        Returns
        -------
        stacked_Xs : container
            Stacked containers.
        """
        raise NotImplementedError("hstack is not implemented for Timeseries")


ADAPTERS_MANAGER.register(BlueAdapter())


class BluePipeline(Pipeline):
    """Pipeline of transforms with a final estimator.

    Wraps the scikit-learn :class:`sklearn.pipeline.Pipeline` class.
    Unique features of the :class:`BluePipeline` class include:
    - Timeseries-aware transformations: The pipeline can handle Timeseries objects.
    """

    def __init__(self, steps, *, memory=None, verbose=False):
        """Pipeline of transforms with a final estimator.

        Wraps the scikit-learn :class:`sklearn.pipeline.Pipeline` class.
        Unique features of the :class:`BluePipeline` class include:
        - Timeseries-aware transformations: The pipeline can handle Timeseries objects.

        Parameters
        ----------
        steps :  list of tuples
            List of (name of step, estimator) tuples that are to be chained in
            sequential order. To be compatible with the scikit-learn API, all steps
            must define `fit`. All non-last steps must also define `transform`. See
            :ref:`Combining Estimators <combining_estimators>` for more details.
        memory : str or object with the joblib.Memory interface, default=None
            Used to cache the fitted transformers of the pipeline.
        verbose : bool, default=False
            If True, the time elapsed while fitting each step will be printed as it
            is completed.
        """
        super().__init__(steps, memory=memory, verbose=verbose)
        # Makes I/O consistent with BlueML
        self.set_output(transform="blueml")
        # Hook on "inverse_transform"  output wrapper
        for _, step in self.steps:
            # Only wrap methods defined by cls itself
            if "inverse_transform" not in step.__class__.__dict__:
                continue
            wrapped_method = _wrap_method_output(
                getattr(step.__class__, "inverse_transform"), "transform"
            )
            setattr(step.__class__, "inverse_transform", wrapped_method)

    @staticmethod
    def read(filename: str) -> "BluePipeline":
        """Read a pipeline from a file."""
        obj = ZipIO.read_zip(filename)["blue_ml_pipeline"]
        if not isinstance(obj, (Pipeline, BluePipeline)):
            raise ValueError(f"Object loaded from {filename} is not a valid Pipeline.")
        return obj

    def write(self, filename: str) -> None:
        """Write a pipeline to a file."""
        ZipIO.write_zip(filename, {"blue_ml_pipeline": self})

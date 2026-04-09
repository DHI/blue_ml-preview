"""ML-specific utilities for Timeseries.

This module handles ML-specific operations like validation and multihorizon targets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from blue_ml.timeseries.timeseries import Timeseries


class MLAdapter:
    """Handles ML-specific operations for Timeseries.

    Responsibility: Provide ML-specific transformations and validations.
    """

    @staticmethod
    def is_valid_for_ml(ts: Timeseries) -> bool:
        """Check if timeseries is valid for ML use.

        Parameters
        ----------
        ts : Timeseries
            Timeseries to validate

        Returns
        -------
        bool
            True if valid for ML, False otherwise
        """
        return ts.report.warnings.is_valid

    @staticmethod
    def set_multihorizon_target(ts: Timeseries, horizon: int) -> Timeseries:
        """Set up multihorizon targets for forecasting.

        Parameters
        ----------
        ts : Timeseries
            Timeseries to process
        horizon : int
            Forecast horizon

        Returns
        -------
        Timeseries
            Timeseries with multihorizon targets
        """
        from blue_ml.timeseries.timeseries import Timeseries
        import numpy as np

        horizon += 1  # Include current timestep

        target_names_list = []
        target_values_list = []

        for target_name in ts.targets.names:
            target_item = ts.targets[target_name]
            target_values = target_item.values

            for h in range(horizon):
                new_name = f"{target_name}_h{h}"
                target_names_list.append(new_name)

                # Shift values by h steps
                if h == 0:
                    shifted_values = target_values
                else:
                    shifted_values = np.roll(target_values, -h)
                    shifted_values[-h:] = np.nan  # Set last h values to NaN

                target_values_list.append(shifted_values)

        # Stack all target arrays
        all_targets = np.column_stack(target_values_list)

        # Create new timeseries with multihorizon targets
        return Timeseries.from_index_and_values(
            values=np.column_stack([ts.features.values, all_targets]),
            index=ts.time,
            column_names=ts.features.names + target_names_list,
            target_names=target_names_list,
            attrs=ts.attrs,
        )

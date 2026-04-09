from __future__ import annotations

from typing import Any, Protocol

import blue_ml
from blue_ml.timeseries.pipeline._pipeline import BluePipeline
from blue_ml.timeseries.timeseries import Timeseries


class TransformLike(Protocol):
    """Protocol for transform objects consumed by ModelFrame."""

    def fit(self, x: Any = None, y: Any = None) -> Any: ...
    def transform(self, x: Any, **params: Any) -> Any: ...
    def fit_transform(self, x: Any, y: Any = None, **params: Any) -> Any: ...


class ScalerLike(TransformLike, Protocol):
    """Protocol for transform objects that also support inverse transforms."""

    def inverse_transform(self, x: Any, **params: Any) -> Any: ...


class ModelLike(Protocol):
    """Protocol for predictive models used by ModelFrame."""

    def fit(self, x: Any, y: Any = None, **params: Any) -> Any: ...
    def predict(self, x: Any, **params: Any) -> Any: ...


class ModelFrame:
    """Coordinate preprocessing, model fitting, prediction, and evaluation.

    The class wraps a predictive model together with optional pre-processing,
    scaling, and post-processing steps. It is intended to provide a compact,
    notebook-friendly workflow for fitting on a training split and evaluating on
    a validation split while preserving the BlueML ``Timeseries`` structure.
    """

    def __init__(
        self,
        name: str,
        model: ModelLike | None = None,
        pre_transforms: BluePipeline | TransformLike | None = None,
        scaler: ScalerLike | None = None,
        post_transforms: BluePipeline | TransformLike | None = None,
    ) -> None:
        """Initialize a model workflow container.

        Parameters
        ----------
        name : str
            Name used when reporting predictions during evaluation.
        model : ModelLike | None, optional
            Predictive model implementing ``fit`` and ``predict``.
        pre_transforms : BluePipeline | TransformLike | None, optional
            Transformations applied before scaling and model fitting.
        scaler : ScalerLike | None, optional
            Scaling transform applied before model fitting and inverted after
            prediction.
        post_transforms : BluePipeline | TransformLike | None, optional
            Transformations applied to model predictions after inverse scaling.
        """
        self.name = name
        self.model = model
        self.scaler = scaler
        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms
        self.ts_val: Timeseries | None = None

    def add_model(self, model: ModelLike) -> ModelFrame:
        """Attach a predictive model to the frame."""
        self.model = model
        return self

    def add_scaler(self, scaler: ScalerLike) -> ModelFrame:
        """Attach a scaler used before fitting and inverted after prediction."""
        self.scaler = scaler
        return self

    def add_pre_transforms(
        self,
        transforms: BluePipeline | TransformLike,
    ) -> ModelFrame:
        """Attach transformations applied before scaling and model fitting."""
        self.pre_transforms = transforms
        return self

    def add_post_transforms(
        self,
        transforms: BluePipeline | TransformLike,
    ) -> ModelFrame:
        """Attach transformations applied after prediction and inverse scaling."""
        self.post_transforms = transforms
        return self

    def fit(self, timeseries: Timeseries, split: float = 0.7, **kwargs: Any) -> None:
        """Fit the configured model on a training split of the input timeseries.

        Parameters
        ----------
        timeseries : Timeseries
            Input timeseries containing features and targets.
        split : float, default=0.7
            Fraction of data used for training. The remainder is stored as the
            validation set.
        **kwargs : Any
            Additional keyword arguments forwarded to ``model.fit``.
        """
        ts_train, ts_val = blue_ml.timeseries.operations.train_test_split(
            timeseries, split
        )
        self.ts_val = ts_val

        if self.pre_transforms:
            ts_train_T = self.pre_transforms.fit_transform(ts_train)
        else:
            ts_train_T = ts_train.copy()

        if self.scaler:
            ts_train_TT = self.scaler.fit_transform(ts_train_T)
        else:
            ts_train_TT = ts_train_T.copy()

        self.model.fit(ts_train_TT, **kwargs)

    def predict(self, timeseries: Timeseries) -> Timeseries:
        """Generate predictions for the provided timeseries.

        Parameters
        ----------
        timeseries : Timeseries
            Input timeseries to transform and pass through the model.

        Returns
        -------
        Timeseries
            Predicted timeseries after inverse scaling and optional
            post-processing.
        """
        if self.pre_transforms:
            ts_features_T = self.pre_transforms.transform(timeseries)
        else:
            ts_features_T = timeseries.copy()

        if self.scaler:
            ts_features_TT = self.scaler.transform(ts_features_T)
        else:
            ts_features_TT = ts_features_T.copy()

        ts_pred_TT = self.model.predict(ts_features_TT)

        if self.scaler:
            ts_pred_T = self.scaler.inverse_transform(ts_pred_TT)
        else:
            ts_pred_T = ts_pred_TT.copy()

        if self.post_transforms:
            ts_pred = self.post_transforms.fit_transform(ts_pred_T)
        else:
            ts_pred = ts_pred_T.copy()

        return ts_pred

    def evaluate(
        self,
        ts_target: Any = None,
        ts_baseline: Any = None,
        quantity: Any = None,
    ) -> blue_ml.ModelSkillAssessor:
        """Evaluate predictions against a target and optional baseline.

        Parameters
        ----------
        ts_target : Any, optional
            Target to evaluate against. If omitted, the single target in the
            stored validation set is used.
        ts_baseline : Any, optional
            Baseline prediction added to the skill assessment.
        quantity : Any, optional
            Quantity metadata forwarded to the model skill assessor.

        Returns
        -------
        blue_ml.ModelSkillAssessor
            Skill assessor populated with target, baseline, and predictions.
        """
        if self.ts_val is None:
            raise ValueError("ModelFrame must be fitted before calling evaluate.")

        if ts_target is None:
            if len(self.ts_val.targets.names) > 1:
                raise ValueError(
                    "Multiple targets found in validation set. Please specify the target to evaluate against."
                )
            elif len(self.ts_val.targets.names) == 0:
                raise ValueError(
                    "No targets found in validation set. Please specify the target to evaluate against."
                )
            ts_target = self.ts_val[self.ts_val.targets.names[0]]

        ts_pred = self.predict(self.ts_val)

        skill = blue_ml.ModelSkillAssessor()
        skill.add_target(ts_target, quantity=quantity)
        if ts_baseline is not None:
            skill.add_baseline(ts_baseline, name="Baseline", quantity=quantity)

        for itm in ts_pred.targets.names:
            skill.add_prediction(ts_pred[itm], name=self.name, quantity=quantity)

        return skill

    def fit_evaluate(
        self,
        timeseries: Timeseries,
        split: float = 0.7,
        ts_baseline: Any = None,
        **kwargs: Any,
    ) -> blue_ml.ModelSkillAssessor:
        """Fit the model and immediately evaluate it on the validation split.

        Parameters
        ----------
        timeseries : Timeseries
            Input timeseries containing features and a single target.
        split : float, default=0.7
            Fraction of data used for training.
        ts_baseline : Any, optional
            Baseline prediction added during evaluation.
        **kwargs : Any
            Additional keyword arguments forwarded to ``fit``.

        Returns
        -------
        blue_ml.ModelSkillAssessor
            Skill assessor for the validation predictions.
        """
        if len(timeseries.targets.names) > 1:
            raise ValueError(
                "fit_evaluate expects a single target in the input timeseries. Please specify the target to evaluate against or reduce the timeseries to a single target."
            )
        self.fit(timeseries, split=split, **kwargs)
        return self.evaluate(ts_baseline=ts_baseline)

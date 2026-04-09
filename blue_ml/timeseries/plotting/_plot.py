from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore
import seaborn as sns  # type: ignore
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from blue_ml.timeseries import Timeseries
from blue_ml.timeseries.timeseries_item import ItemAlias


class _Plotter:
    """Plotter class for Timeseries."""

    def __init__(self, ts: Timeseries):
        self.ts = ts

    def __call__(self, group_by_attr=None, figsize=None):
        self.plot(group_by_attr=group_by_attr, figsize=figsize)

    def plot(self, group_by_attr=None, figsize=None, include_empties=True):
        """Plot all items as timeseries."""
        ts = self.ts

        if group_by_attr is None:
            if figsize is None:
                figsize = (8, 4)
            fig, ax = plt.subplots()

            for tsi in ts.features:
                ax.plot(tsi.index, tsi.values, label=tsi.name)

            for tsi in ts.targets:
                ax.plot(tsi.index, tsi.values, label=tsi.name)

            ax.legend()

            return ax

        else:
            unique_attrs_split = {}
            for label in ["features", "targets"]:
                # For each item, check if attr is in item
                # If it is, add to dict with key as attr value
                unique_attrs = {}
                ts_subset = getattr(ts, label)
                for v in ts_subset.names:
                    if group_by_attr in ts_subset._attrs[v].keys():
                        k = ts_subset._attrs[v][group_by_attr]
                        if k not in unique_attrs.keys():
                            unique_attrs[k] = [v]
                        else:
                            unique_attrs[k].append(v)
                    else:
                        if include_empties:
                            if "other" not in unique_attrs.keys():
                                unique_attrs["other"] = [v]
                            else:
                                unique_attrs["other"].append(v)
                unique_attrs_split[label] = unique_attrs

            n_plot_features = len(unique_attrs_split["features"].keys())
            n_plot_targets = len(unique_attrs_split["targets"].keys())
            n_plot_items = n_plot_features + n_plot_targets

            fig, axs = plt.subplots(n_plot_items, 1, figsize=(10, 10), sharex=True)

            for i, (attr, items) in enumerate(unique_attrs_split["features"].items()):
                axs[i].set_title("Features | " + str(attr))
                for itm in items:
                    axs[i].plot(ts[itm].time, ts[itm].values, label=itm)
                axs[i].legend()

            for i, (attr, items) in enumerate(unique_attrs_split["targets"].items()):
                j = i + n_plot_features
                axs[j].set_title("Targets | " + str(attr))
                for itm in items:
                    axs[j].plot(ts[itm].time, ts[itm].values, label=itm)
                axs[j].legend()
            # Get unique attribute values from both features and targets
            unique_groups = set()
            for tsi in ts.features:
                if group_by_attr in tsi._attrs:
                    unique_groups.add(tsi._attrs[group_by_attr])
            for tsi in ts.targets:
                if group_by_attr in tsi._attrs:
                    unique_groups.add(tsi._attrs[group_by_attr])
            unique_groups = list(unique_groups)

            valid_groups_features = [
                ts.sel_from_attrs(**{group_by_attr: ug}).n_features != 0
                for ug in unique_groups
            ]
            valid_group_targets = [
                ts.sel_from_attrs(**{group_by_attr: ug}).n_targets != 0
                for ug in unique_groups
            ]
            n_features = np.sum(valid_groups_features)
            n_targets = np.sum(valid_group_targets)
            n_items = n_features + n_targets

            # Init plot
            if figsize is None:
                figsize = (8, 3 * n_items)
            fig, axs = plt.subplots(nrows=n_items, figsize=figsize, sharex=True)

            c = 0
            for i, (g, cond) in enumerate(zip(unique_groups, valid_groups_features)):
                if cond:
                    ts_subset = ts.sel_from_attrs(**{group_by_attr: g})
                    for tsi in ts_subset.features:
                        x = tsi.time
                        y = tsi.values
                        axs[c].plot(x, y, label=tsi.name)
                    axs[c].legend()
                    axs[c].set_title("Features: " + str(g))
                    c += 1

            for i, (g, cond) in enumerate(zip(unique_groups, valid_group_targets)):
                if cond:
                    ts_subset = ts.sel_from_attrs(**{group_by_attr: g})
                    for tsi in ts_subset.targets:
                        x = tsi.time
                        y = tsi.values
                        axs[c].plot(x, y, label=tsi.name)
                    axs[c].legend()
                    axs[c].set_title("targets: " + str(g))
                    c += 1

    def time(self, ax=None, group_by_attr=None):
        ts = self.ts
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        plt_sts = dict(solid_capstyle="butt", lw=1)
        plt_sts_features = dict(c="r")
        plt_sts_targets = dict(c="b")

        if group_by_attr is None:
            n_features = ts.n_features
            n_targets = ts.n_targets
            n_items = n_features + n_targets
            names = ts.features.names + ts.targets.names

            c = 0
            for i, tsi in enumerate(ts.features):
                time = tsi.time
                values = ~np.isnan(np.asarray(tsi.values)) * c
                ax.plot(time, values, **plt_sts_features, **plt_sts)
                c += 1

            for i, tsi in enumerate(ts.targets):
                time = tsi.time
                values = ~np.isnan(np.asarray(tsi.values)) * c
                ax.plot(time, values, **plt_sts_targets, **plt_sts)
                c += 1

        else:
            # Get unique attribute values from both features and targets
            unique_groups = set()
            for tsi in ts.features:
                if group_by_attr in tsi._attrs:
                    unique_groups.add(tsi._attrs[group_by_attr])
            for tsi in ts.targets:
                if group_by_attr in tsi._attrs:
                    unique_groups.add(tsi._attrs[group_by_attr])
            unique_groups = list(unique_groups)

            valid_groups_features = [
                ts.sel_from_attrs(**{group_by_attr: ug}).n_features != 0
                for ug in unique_groups
            ]
            valid_group_targets = [
                ts.sel_from_attrs(**{group_by_attr: ug}).n_targets != 0
                for ug in unique_groups
            ]

            n_features = np.sum(valid_groups_features)
            n_targets = np.sum(valid_group_targets)
            n_items = n_features + n_targets
            names = [i for i, j in zip(unique_groups, valid_groups_features) if j] + [
                i for i, j in zip(unique_groups, valid_group_targets) if j
            ]

            c = 0
            for i, (g, cond) in enumerate(zip(unique_groups, valid_groups_features)):
                if cond:
                    ts_subset = ts.sel_from_attrs(**{group_by_attr: g})
                    time = ts_subset.features.time
                    values = ~np.isnan(np.asarray(ts_subset.features.values))[:, 0] * c
                    ax.plot(time, values, **plt_sts_features, **plt_sts)
                    c += 1

            for i, (g, cond) in enumerate(zip(unique_groups, valid_group_targets)):
                if cond:
                    ts_subset = ts.sel_from_attrs(**{group_by_attr: g})
                    time = ts_subset.targets.time
                    values = ~np.isnan(np.asarray(ts_subset.targets.values))[:, 0] * c
                    ax.plot(time, values, **plt_sts_targets, **plt_sts)
                    c += 1

        # Dummy plots for legend
        ax.plot(
            [np.nan],
            [np.nan],
            target="Feature (n={})".format(n_features),
            **plt_sts_features,
        )
        ax.plot(
            [np.nan],
            [np.nan],
            target="target (n={})".format(n_targets),
            **plt_sts_targets,
        )

        # Plot time available
        time_in = ts.time

        ax.fill_betweenx(
            [-1.5, n_items + 0.5],
            time_in[0],
            time_in[-1],
            facecolor="k",
            alpha=0.2,
            zorder=-1,
        )
        ax.plot(
            time_in,
            [-1] * len(time_in),
            c="k",
            lw=2,
            solid_capstyle="butt",
            label="Time, joint",
        )

        # Format plot
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1))
        ax.set_ylim(-1.5, n_items - 0.5)

        ax.invert_yaxis()
        ax.set_yticks(np.arange(-1, len(names)))
        ax.set_yticktargets(["Time available"] + names)

        ax.grid(axis="x")
        ax.grid(axis="y", ls=":")

    def gaps(
        self, min_gap_size: int = 5, backend: Literal["plotly", "matplotlib"] = "plotly"
    ) -> Union[go.Figure, Figure]:
        color_gaps = "tomato"
        color_background = "seagreen"
        color_aux = "lightgrey"

        ini_time, end_time = self.ts.time.min(), self.ts.time.max()
        features_gaps, targets_gaps = self.ts.gaps()
        # Combine gaps from both features and targets
        gaps = pd.concat([features_gaps, targets_gaps], ignore_index=True)
        gaps = gaps[gaps["gap_size"] >= min_gap_size].copy()

        if backend == "plotly":
            # Plotly requires traces to be able to render vrect
            two_dummy_points = [0.5, 0.5]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=[ini_time, end_time], y=two_dummy_points, hoverinfo="skip")
            )
            fig.add_vrect(
                x0=ini_time, x1=end_time, fillcolor=color_background, line=dict(width=0)
            )

            for i, gap in gaps.iterrows():
                fig.add_trace(
                    go.Scatter(
                        x=[gap["gap_start"], gap["gap_end"]],
                        y=two_dummy_points,
                        name="",
                        line=dict(color=color_gaps),
                        marker=dict(color=color_gaps),
                        hovertemplate="%{x}",
                    )
                )
                fig.add_vrect(
                    x0=gap["gap_start"],
                    x1=gap["gap_end"],
                    fillcolor=color_gaps,
                    line_width=0,
                )

            fig.update_layout(
                xaxis=dict(
                    tickmode="array",
                    range=(ini_time, end_time),
                    showspikes=True,
                    spikemode="across",
                    spikesnap="cursor",
                    showline=True,
                ),
                yaxis=dict(showticklabels=False, showgrid=False),
                height=150,
                margin=dict(l=20, r=20, t=20, b=40),
                showlegend=False,
                hovermode="x",
            )

        else:
            is_ini_year = (
                (self.ts.time.month == 1)
                & (self.ts.time.day == 1)
                & (self.ts.time.hour == 0)
            )
            ini_year = self.ts.time[is_ini_year]

            fig, ax = plt.subplots(1, 1, figsize=(20, 2))
            ax.axvspan(
                xmin=mdates.date2num(ini_time),
                xmax=mdates.date2num(end_time),
                facecolor=color_background,
            )
            for i, gap in gaps.iterrows():
                ax.axvspan(
                    xmin=gap["gap_start"], xmax=gap["gap_end"], facecolor=color_gaps
                )

            ax.set_xticks(ini_year)
            ax.set_xticklabels(
                [str(year) for year in ini_year.year],
                va="center",
                ha="right",
                y=0.5,
                color=color_aux,
            )
            ax.tick_params(left=False, right=False, labelleft=False, rotation=90)

            ax.grid(visible=False, axis="y")
            ax.grid(which="major", axis="x", linestyle="--", color=color_aux)
            ax.set_xlim(mdates.date2num(ini_time), mdates.date2num(end_time))
            for pos in ["top", "bottom", "left", "right"]:
                ax.spines[pos].set_visible(True)

        return fig

    def target_comparison(self, other_model: str):
        if len(self.ts.targets.names) > 1:
            raise NotImplementedError(
                "This method is only valid for Timeseries with one single target"
            )
        else:
            target_name = self.ts.targets.names[0]
            alias_other = ItemAlias(
                variable=self.ts[target_name].variable, source=other_model
            )
            if alias_other.name in self.ts.names:
                columns = [
                    target_name,
                    alias_other.name,
                ]

                features = self.ts.features.to_dataframe()
                targets = self.ts.targets.to_dataframe()
                df = features.merge(targets, left_index=True, right_index=True)[columns]
                # Rename columns to use source names
                column_mapping = {col: self.ts[col].source for col in columns}
                df = df.rename(columns=column_mapping)

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                hp = sns.histplot(
                    data=df.melt(var_name="Model"),
                    x="value",
                    hue="Model",
                    bins=np.linspace(0, 5, 40),
                    kde=True,
                    ax=axes[0],
                )
                sns.regplot(
                    data=df,
                    x=self.ts[target_name].source,
                    y=other_model,
                    lowess=True,
                    scatter_kws=dict(alpha=0.05, color="dodgerblue"),
                    line_kws=dict(linewidth=3, color="blue"),
                    ax=axes[1],
                )
                sns.move_legend(hp, "upper right", frameon=False)
                axes[1].axline((0, 0), (1, 1), linestyle="--", c="black")
                axes[1].set_xlabel(self.ts[target_name].source)
                axes[1].set_ylabel(other_model)
                axes[0].grid(False)
                axes[1].grid(False)
                fig.suptitle(self.ts[target_name].variable)

                plt.tight_layout()
                return fig
            else:
                raise ValueError(
                    f"Invalid other model name: {alias_other.name} was not found in timeseries"
                )

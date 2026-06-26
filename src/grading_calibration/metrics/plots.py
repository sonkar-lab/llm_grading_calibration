"""
Plotting functions for evaluation metrics and feature analysis.
"""

from grading_calibration.domain.models import PredictionResult
from .calibration_bins import compute_calibration_bins, CalibrationBinData
from typing import Any, List, Optional, Literal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox
import glob
import os
from pathlib import Path


def find_plot_folder_from_root(root_dir: str) -> str:
    # Walk up until we find grading_calibration
    current = Path.cwd()

    while not os.path.basename(current) == root_dir:
        parent = os.path.dirname(current)
        if parent == current:  # At filesystem root, not found
            return []
        current = parent

    # Now search down from grading_calibration
    return Path(f"{current}/experiments/plots/")


def plot_calibration_curve(
    predictions: list[PredictionResult],
    num_bins: int = 10,  # CHANGED: from 50 to 10 to standardize with ECE
    use_calibrated: bool = False,
    bin_data: Optional[CalibrationBinData] = None,  # NEW: allow pre-computed bin data
    name: str = "calibration_curve",
    model_name: str = "",
    method_name: str = "",
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Plot calibration curve showing the relationship between confidence and accuracy.

    This function now uses the unified binning logic from compute_calibration_bins()
    to ensure consistency with ECE computation. The x-axis uses the ACTUAL mean
    confidence in each bin (not bin centers).

    Args:
        predictions: List of PredictionResult objects
        num_bins: Number of bins (default 10, matching ECE standard)
        use_calibrated: If True, use calibrated_confidence; else raw_confidence
        bin_data: Optional pre-computed CalibrationBinData. If provided, num_bins is ignored.
        name: Base filename for saved plots
        model_name: Model name for plot title
        method_name: Method name for subfolder organization

    Returns:
        Dictionary with keys:
            - "bin_data": CalibrationBinData object with all bin statistics
            - "ece": Expected Calibration Error computed from bins
    """
    # Compute bin data if not provided
    if bin_data is None:
        bin_data = compute_calibration_bins(predictions, num_bins, use_calibrated)

    plot_folder = Path(output_dir) if output_dir else (find_plot_folder_from_root("grading_calibration") / Path(method_name))
    if not plot_folder.exists():
        os.makedirs(plot_folder)

    if not plot_folder.exists():
        raise FileNotFoundError("Plot folder not found.")

    # Pick a non-clobbering filename stem if the target already exists.
    def _unique_plot_stem(base: str) -> str:
        candidate = base
        if not (plot_folder / f"{candidate}.png").exists() and not (plot_folder / f"{candidate}_histogram.png").exists():
            return candidate
        idx = 2
        while (plot_folder / f"{base}_v{idx}.png").exists() or (plot_folder / f"{base}_v{idx}_histogram.png").exists():
            idx += 1
        return f"{base}_v{idx}"

    plot_stem = _unique_plot_stem(name)
    plot_path = plot_folder / f"{plot_stem}.png"
    hist_path = plot_folder / f"{plot_stem}_histogram.png"
    # Plot calibration curve - use MEAN CONFIDENCE as x-axis (true calibration curve)
    curve_figsize = (10, 8)
    hist_figsize = curve_figsize
    title_size = 32
    label_size = 28
    tick_size = 24
    legend_size = 20
    annotation_size = 22
    bar_label_size = 24

    plt.figure(figsize=curve_figsize)
    plt.plot(
        bin_data.mean_confidences,  # CHANGED: was bin_centers
        bin_data.accuracies,
        marker="o",
        linewidth=4,
        markersize=12,
        label="Calibration curve"
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    # Annotate each point with sample size, accuracy, and ACTUAL mean confidence.
    # Keep labels close to points; apply small nudges only when overlaps occur.
    annotations = []
    placed_bboxes = []
    fig = plt.gcf()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = plt.gca()
    markersize = 8
    # Treat each plotted point as a small "no-cover" box in display coords.
    marker_radius = markersize * fig.dpi / 72.0 * 1.2
    point_bboxes = []
    for mean_conf, acc in zip(bin_data.mean_confidences, bin_data.accuracies):
        x_disp, y_disp = ax.transData.transform((mean_conf, acc))
        point_bboxes.append(
            Bbox.from_extents(
                x_disp - marker_radius,
                y_disp - marker_radius,
                x_disp + marker_radius,
                y_disp + marker_radius,
            )
        )
    base_offset = 1
    offset_step = 1
    max_offset = 34
    bbox_pad = 0.2
    dy_bias = -18
    left_extra = -6
    for i, (mean_conf, acc, size) in enumerate(
        zip(bin_data.mean_confidences, bin_data.accuracies, bin_data.bin_sizes)
    ):
        annotation = f"n={size}\nConf={mean_conf:.3f}\nAcc={acc:.3f}"
        # Place labels close to points with a fixed pattern:
        # alternate up-left / down-right, and force the last bin to down-right.
        if i == len(bin_data.mean_confidences) - 1:
            dx_sign, dy_sign = (1, -1)
        else:
            dx_sign, dy_sign = (-1, 0.8) if (len(bin_data.mean_confidences) - 1 - i) % 2 == 1 else (1, -0.8)
        # If accuracy is 0 and the default is down-right, flip to up-right.
        if acc == 0 and dx_sign == 1 and dy_sign < 0:
            dy_sign = 1
        mag = base_offset
        dx_scale = 5.0 if dx_sign < 0 else 1.0
        extra = left_extra if dx_sign < 0 else 0
        extra_down = 0 if acc == 0 else (dy_bias + extra)
        dx, dy = (dx_sign * mag * dx_scale, dy_sign * mag + extra_down)
        ann = plt.annotate(
            annotation,
            xy=(mean_conf, acc),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=annotation_size,
            bbox=dict(boxstyle=f"round,pad={bbox_pad}", facecolor="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", lw=0.5),
            annotation_clip=False,
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = ann.get_window_extent(renderer).expanded(1.0, 1.0)

        def _overlaps_points(bbox_to_check) -> bool:
            return any(bbox_to_check.overlaps(pbox) for pbox in point_bboxes)

        while (
            (any(bbox.overlaps(existing) for existing in placed_bboxes) or _overlaps_points(bbox))
            and mag < max_offset
        ):
            mag += offset_step
            extra = left_extra if dx_sign < 0 else 0
            extra_down = 0 if acc == 0 else (dy_bias + extra)
            dx, dy = (dx_sign * mag * dx_scale, dy_sign * mag + extra_down)
            ann.set_position((dx, dy))
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = ann.get_window_extent(renderer).expanded(1.0, 1.0)
        placed_bboxes.append(bbox)
        annotations.append(ann)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Mean Confidence", fontsize=label_size)  # CHANGED: clarified label
    plt.ylabel("Accuracy", fontsize=label_size)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.8)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=legend_size)
    title = f" ({method_name})" if method_name else ""
    title += (
        f"Calibration Curve for {model_name}" if model_name else "Calibration Curve"
    )
    plt.title(title, fontsize=title_size, y=1.03)
    # Save the plot
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    # plt.show()

    plt.close()

    # Plot histogram of bin sizes - use bin centers for x-axis (evenly spaced)
    plt.figure(figsize=hist_figsize)
    bars = plt.bar(
        bin_data.bin_centers,
        bin_data.bin_sizes,
        width=1 / len(bin_data.bin_boundaries),
        edgecolor="black",
        label="Prediction Count",
    )
    max_count = max(bin_data.bin_sizes) if len(bin_data.bin_sizes) > 0 else 0
    plt.ylim(0, max_count * 1.1 + 1)

    # Add count labels on top of each bar
    for bar, count in zip(bars, bin_data.bin_sizes):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=bar_label_size,
        )

    plt.xlim(0, 1)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.8)
    plt.xlabel("Confidence Bin", fontsize=label_size)
    plt.ylabel("Number of Predictions", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    title = f" ({method_name})" if method_name else ""
    title += (
        f"Confidence Histogram for {model_name}"
        if model_name
        else "Confidence Histogram"
    )
    plt.title(title, fontsize=title_size, y=1.03)
    plt.grid()
    # Save the histogram
    plt.legend(fontsize=legend_size)
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    # plt.show()
    plt.close()

    # Return bin data and ECE for callers who need detailed statistics
    return {"bin_data": bin_data, "ece": bin_data.get_ece(), "plot_path": plot_path, "hist_path": hist_path, "plot_stem": plot_stem}


def plot_feature_correlation_heatmap(
    features_df: pd.DataFrame,
    target_columns: List[str] = ['raw_confidence', 'is_correct'],
    feature_columns: Optional[List[str]] = None,
    name: str = "correlation_heatmap",
    method_name: str = "",
    output_dir: Optional[Path] = None,
    figsize: tuple = (12, 10),
) -> None:
    """
    Plot heatmap of feature correlations with target variables.

    Args:
        features_df: DataFrame with extracted features
        target_columns: List of target variable column names
        feature_columns: List of feature columns (if None, auto-select numerical)
        name: Filename for saving
        method_name: Method name for folder organization
        figsize: Figure size tuple
    """
    import seaborn as sns

    # Get feature columns
    if feature_columns is None:
        exclude_cols = ['raw_confidence', 'calibrated_confidence', 'confidence_delta',
                        'is_correct', 'prediction', 'ground_truth', 'confidence_bin',
                        'is_high_confidence', 'is_medium_confidence', 'is_low_confidence']
        feature_columns = [c for c in features_df.select_dtypes(include=[np.number]).columns
                           if c not in exclude_cols]

    # Compute correlations with targets
    corr_data = {}
    for target in target_columns:
        if target in features_df.columns:
            corr_data[target] = features_df[feature_columns].corrwith(features_df[target])

    if not corr_data:
        print("No valid target columns found")
        return

    corr_df = pd.DataFrame(corr_data)

    # Sort by absolute correlation with first target
    corr_df['abs_sort'] = corr_df[target_columns[0]].abs()
    corr_df = corr_df.sort_values('abs_sort', ascending=False).drop('abs_sort', axis=1)

    # Take top features
    corr_df = corr_df.head(20)

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_df, annot=True, cmap='RdBu_r', center=0, fmt='.3f',
                vmin=-1, vmax=1, linewidths=0.5)
    plt.title(f"Feature Correlations with Targets ({method_name})" if method_name else "Feature Correlations")
    plt.tight_layout()

    # Save
    plot_folder = output_dir or (find_plot_folder_from_root("grading_calibration") / Path(method_name))
    plot_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{plot_folder}/{name}.png", dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_mixed_model_effects_heatmap(
    effects_df: pd.DataFrame,
    name: str = "mixed_model_effects_heatmap",
    method_name: str = "",
    output_dir: Optional[Path] = None,
    max_features: int = 30,
    effect_scale: Literal["beta", "signed_log_or"] = "beta",
    figsize: tuple = (12, 10),
) -> None:
    """
    Plot a heatmap of mixed-model feature effects across targets.

    Expects the CSV output from step3_feature_analysis.py --mixed-model-effects.
    For binary outcomes, the heatmap uses the fixed-effect slope `beta` by default.
    """
    import seaborn as sns

    if effects_df.empty:
        print("No mixed-model effects to plot")
        return

    df = effects_df.copy()
    if "error" in df.columns:
        df = df[df["error"].isna()]

    required = {"feature", "target", "beta", "outcome_type"}
    missing = required - set(df.columns)
    if missing:
        print(f"Mixed-model effects missing required columns: {sorted(missing)}")
        return

    df = df[np.isfinite(df["beta"].astype(float))]
    if df.empty:
        print("No finite mixed-model beta values to plot")
        return

    df["beta"] = df["beta"].astype(float)

    if effect_scale == "signed_log_or":
        df["effect_value"] = df["beta"]
        df["effect_label"] = "Signed log-odds slope (beta)"
    else:
        df["effect_value"] = df["beta"]
        df["effect_label"] = "Fixed-effect slope (beta)"

    pivot = df.pivot_table(index="feature", columns="target", values="effect_value", aggfunc="first")
    if pivot.empty:
        print("No mixed-model effects after pivoting")
        return

    max_abs = pivot.abs().max(axis=1).sort_values(ascending=False)
    top = max_abs.head(max_features).index
    pivot = pivot.loc[top]

    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot,
        annot=True,
        cmap="RdBu_r",
        center=0,
        fmt=".3f",
        linewidths=0.5,
    )
    plt.title(f"Mixed-model feature effects ({method_name})" if method_name else "Mixed-model feature effects")
    plt.ylabel("Feature")
    plt.tight_layout()

    plot_folder = output_dir or (find_plot_folder_from_root("grading_calibration") / Path(method_name))
    plot_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{plot_folder}/{name}.png", dpi=150, bbox_inches="tight")
    # plt.show()
    plt.close()


def plot_mixed_model_effects_forest(
    effects_df: pd.DataFrame,
    name_prefix: str = "mixed_model_effects_forest",
    method_name: str = "",
    output_dir: Optional[Path] = None,
    max_features_per_target: int = 25,
    figsize: Optional[tuple] = None,
) -> None:
    """
    Plot forest plots of mixed-model effects, one figure per target.

    - Continuous outcomes: plots beta with CI, reference line at 0.
    - Binary outcomes: plots odds ratio with CI on log scale, reference line at 1.
    """
    if effects_df.empty:
        print("No mixed-model effects to plot")
        return

    df = effects_df.copy()
    if "error" in df.columns:
        df = df[df["error"].isna()]

    required = {"feature", "target", "outcome_type", "beta", "ci_low", "ci_high"}
    missing = required - set(df.columns)
    if missing:
        print(f"Mixed-model effects missing required columns: {sorted(missing)}")
        return

    for col in ["beta", "ci_low", "ci_high"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[np.isfinite(df["beta"]) & np.isfinite(df["ci_low"]) & np.isfinite(df["ci_high"])]
    if df.empty:
        print("No finite mixed-model effects to plot")
        return

    plot_folder = output_dir or (find_plot_folder_from_root("grading_calibration") / Path(method_name))
    plot_folder.mkdir(parents=True, exist_ok=True)

    for target_label in df["target"].dropna().unique():
        sub = df[df["target"] == target_label].copy()
        if sub.empty:
            continue

        sub["abs_beta"] = sub["beta"].abs()
        sub = sub.sort_values("abs_beta", ascending=False).head(max_features_per_target)
        sub = sub.sort_values("abs_beta", ascending=True)

        outcome_types = set(sub["outcome_type"].dropna().unique())
        is_binary = outcome_types == {"binary"} or ("binary" in outcome_types and len(outcome_types) == 1)

        n = len(sub)
        if n == 0:
            continue

        if figsize is None:
            fig_h = max(4.0, 0.35 * n + 2.0)
            fig_w = 10.0
            fig_size = (fig_w, fig_h)
        else:
            fig_size = figsize

        y = np.arange(n)
        labels = sub["feature"].astype(str).tolist()

        plt.figure(figsize=fig_size)
        if is_binary:
            # Plot odds ratios derived from beta (log-odds slope).
            or_mid = np.exp(sub["beta"].to_numpy())
            or_lo = np.exp(sub["ci_low"].to_numpy())
            or_hi = np.exp(sub["ci_high"].to_numpy())

            plt.hlines(y, or_lo, or_hi, color="0.3", linewidth=2)
            plt.plot(or_mid, y, "o", color="0.1", markersize=5)
            plt.axvline(1.0, color="0.6", linestyle="--", linewidth=1)
            plt.xscale("log")
            plt.xlabel("Odds ratio (per 1 SD increase in feature)")
            plt.title(f"Mixed-model effects (GLMM VB, screening): {target_label}")
        else:
            beta = sub["beta"].to_numpy()
            lo = sub["ci_low"].to_numpy()
            hi = sub["ci_high"].to_numpy()

            plt.hlines(y, lo, hi, color="0.3", linewidth=2)
            plt.plot(beta, y, "o", color="0.1", markersize=5)
            plt.axvline(0.0, color="0.6", linestyle="--", linewidth=1)
            plt.xlabel("Beta (per 1 SD increase in feature)")
            plt.title(f"Mixed-model effects (LMM): {target_label}")

        plt.yticks(y, labels)
        plt.tight_layout()

        out_path = plot_folder / f"{name_prefix}__{target_label.replace(' ', '_')}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        # plt.show()
        plt.close()


def plot_confidence_by_category(
    features_df: pd.DataFrame,
    category_column: str,
    name: str = "confidence_by_category",
    method_name: str = "",
    figsize: tuple = (12, 6),
    max_categories: int = 20,
) -> None:
    """
    Plot box plots of confidence distribution per category (rubric_id, question_id).

    Args:
        features_df: DataFrame with extracted features
        category_column: Column to group by (e.g., 'rubric_id', 'question_id')
        name: Filename for saving
        method_name: Method name for folder organization
        figsize: Figure size tuple
        max_categories: Maximum number of categories to show
    """
    if category_column not in features_df.columns:
        print(f"Column {category_column} not found")
        return

    # Get category counts and select top categories
    category_counts = features_df[category_column].value_counts()
    top_categories = category_counts.head(max_categories).index.tolist()
    
    df_filtered = features_df[features_df[category_column].isin(top_categories)]

    plt.figure(figsize=figsize)
    
    # Create box plot
    categories = df_filtered[category_column].unique()
    data = [df_filtered[df_filtered[category_column] == cat]['raw_confidence'].values 
            for cat in categories]
    
    bp = plt.boxplot(data, labels=categories, patch_artist=True)
    
    # Color by mean accuracy
    accuracies = [df_filtered[df_filtered[category_column] == cat]['is_correct'].mean() 
                  for cat in categories]
    colors = plt.cm.RdYlGn(accuracies)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.xlabel(category_column)
    plt.ylabel('Raw Confidence')
    plt.title(f"Confidence Distribution by {category_column} ({method_name})" if method_name 
              else f"Confidence Distribution by {category_column}")
    plt.xticks(rotation=45, ha='right')
    
    # Add colorbar for accuracy
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Accuracy')
    
    plt.tight_layout()

    # Save
    plot_folder = find_plot_folder_from_root("grading_calibration") / Path(method_name)
    plot_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{plot_folder}/{name}.png", dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_metric_by_feature_bins(
    features_df: pd.DataFrame,
    feature: str,
    metric: str = 'ece',
    n_bins: int = 5,
    name: Optional[str] = None,
    method_name: str = "",
    figsize: tuple = (10, 6),
) -> dict:
    """
    Plot a metric (ECE, accuracy) vs binned feature values.

    Args:
        features_df: DataFrame with extracted features
        feature: Feature column to bin (e.g., 'answer_length')
        metric: Metric to compute ('ece', 'accuracy', 'brier')
        n_bins: Number of bins
        name: Filename for saving (auto-generated if None)
        method_name: Method name for folder organization
        figsize: Figure size tuple

    Returns:
        Dict with bin centers and metric values
    """
    if feature not in features_df.columns:
        print(f"Feature {feature} not found")
        return {}

    # Create bins
    try:
        features_df['_bin'] = pd.qcut(features_df[feature], q=n_bins, labels=False, duplicates='drop')
        bin_edges = pd.qcut(features_df[feature], q=n_bins, retbins=True, duplicates='drop')[1]
    except ValueError:
        features_df['_bin'] = pd.cut(features_df[feature], bins=n_bins, labels=False)
        bin_edges = pd.cut(features_df[feature], bins=n_bins, retbins=True)[1]

    # Compute metric per bin
    results = []
    for bin_idx in sorted(features_df['_bin'].dropna().unique()):
        bin_data = features_df[features_df['_bin'] == bin_idx]
        
        if len(bin_data) < 2:
            continue
            
        bin_center = (bin_edges[int(bin_idx)] + bin_edges[int(bin_idx) + 1]) / 2
        
        if metric == 'ece':
            # Compute ECE for this bin
            confidences = bin_data['raw_confidence'].values
            correctness = bin_data['is_correct'].values
            metric_value = np.mean(np.abs(confidences - correctness))
        elif metric == 'accuracy':
            metric_value = bin_data['is_correct'].mean()
        elif metric == 'brier':
            confidences = bin_data['raw_confidence'].values
            correctness = bin_data['is_correct'].values
            metric_value = np.mean((confidences - correctness) ** 2)
        else:
            metric_value = bin_data[metric].mean() if metric in bin_data.columns else np.nan
            
        results.append({
            'bin_center': bin_center,
            'bin_idx': bin_idx,
            'metric_value': metric_value,
            'n_samples': len(bin_data),
            'bin_label': f"{bin_edges[int(bin_idx)]:.0f}-{bin_edges[int(bin_idx)+1]:.0f}"
        })

    features_df.drop('_bin', axis=1, inplace=True)

    if not results:
        return {}

    results_df = pd.DataFrame(results)

    # Plot
    plt.figure(figsize=figsize)
    
    # Bar plot with sample sizes
    bars = plt.bar(range(len(results_df)), results_df['metric_value'], 
                   color='steelblue', alpha=0.7, edgecolor='black')
    
    # Add sample size labels
    for i, (bar, n) in enumerate(zip(bars, results_df['n_samples'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                 f'n={n}', ha='center', va='bottom', fontsize=9)

    plt.xticks(range(len(results_df)), results_df['bin_label'], rotation=45, ha='right')
    plt.xlabel(f'{feature} bins')
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} by {feature} ({method_name})" if method_name 
              else f"{metric.upper()} by {feature}")
    plt.tight_layout()

    # Save
    if name is None:
        name = f"{metric}_by_{feature}"
    plot_folder = find_plot_folder_from_root("grading_calibration") / Path(method_name)
    plot_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{plot_folder}/{name}.png", dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()

    return {'results': results_df.to_dict('records')}


def plot_feature_scatter(
    features_df: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    color_by: str = 'is_correct',
    name: Optional[str] = None,
    method_name: str = "",
    figsize: tuple = (10, 8),
    alpha: float = 0.6,
) -> None:
    """
    Create scatter plot of two features with color coding.

    Args:
        features_df: DataFrame with extracted features
        x_feature: Feature for x-axis
        y_feature: Feature for y-axis
        color_by: Feature to use for color coding
        name: Filename for saving (auto-generated if None)
        method_name: Method name for folder organization
        figsize: Figure size tuple
        alpha: Transparency for points
    """
    for col in [x_feature, y_feature, color_by]:
        if col not in features_df.columns:
            print(f"Column {col} not found")
            return

    plt.figure(figsize=figsize)
    
    scatter = plt.scatter(features_df[x_feature], features_df[y_feature],
                          c=features_df[color_by], cmap='RdYlGn',
                          alpha=alpha, edgecolors='none', s=50)
    
    plt.colorbar(scatter, label=color_by)
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title(f"{x_feature} vs {y_feature} (colored by {color_by})")
    
    # Add trend line
    z = np.polyfit(features_df[x_feature], features_df[y_feature], 1)
    p = np.poly1d(z)
    x_line = np.linspace(features_df[x_feature].min(), features_df[x_feature].max(), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.8, label=f"Trend (slope={z[0]:.3f})")
    plt.legend()
    
    plt.tight_layout()

    # Save
    if name is None:
        name = f"scatter_{x_feature}_vs_{y_feature}"
    plot_folder = find_plot_folder_from_root("grading_calibration") / Path(method_name)
    plot_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{plot_folder}/{name}.png", dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()

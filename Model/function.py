"""
================================================================================
 function.py — Shared Utility Functions for Air Quality ML Pipeline
================================================================================

 This module contains all reusable helper functions and custom transformers
 used across the project's EDA, feature engineering, and modelling notebooks.

 Sections
 --------
   1.  Visualisation          — Correlation heatmap helpers
   2.  Feature Engineering    — Angle aggregation, AMF, atmospheric indices
   3.  Column Management      — Generic drop-by-keyword utility
   4.  Collinearity Reduction — Cloud fraction & sensor altitude compression
   5.  Imputation             — Full pipeline imputer + sklearn-compatible class
   6.  Result Plots           — Model comparison, actual vs predicted, residuals

 Dependencies
 ------------
   numpy, pandas, matplotlib, seaborn, scikit-learn

================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin


# ================================================================================
# 1. VISUALISATION
# ================================================================================

def plot_target_correlation(data: pd.DataFrame, target_col: str, title: str = "Feature Correlation with Target"):
    """
    Plot a sorted single-column heatmap of feature correlations with the target.

    Computes Pearson correlation between all numeric features and the specified
    target column, then renders it as a colour-coded heatmap sorted from most
    positive to most negative correlation.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing features and the target column.
    target_col : str
        Name of the target column (e.g. ``'target'``).
    title : str, optional
        Chart title prefix. Defaults to ``'Feature Correlation with Target'``.

    Returns
    -------
    None
        Displays the heatmap inline via ``plt.show()``.

    Example
    -------
    >>> plot_target_correlation(df, target_col='target')
    """
    # Compute correlation for numeric columns only, keeping only the target column
    correlations = data.corr(numeric_only=True)[[target_col]]
    correlations = correlations.sort_values(by=target_col, ascending=False)
    correlations = correlations.drop(target_col)

    plt.figure(figsize=(15, 18))
    sns.heatmap(
        correlations,
        annot=True,
        cmap='viridis',
        vmin=-1,
        vmax=1,
        fmt=".2f",
        linewidths=0.5
    )
    plt.title(f"{title}: {target_col}")
    plt.show()


# ================================================================================
# 2. FEATURE ENGINEERING
# ================================================================================

def relative_mean_angles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate redundant per-band satellite angle columns into two relative features.

    Each satellite band (NO2, O3, CO, etc.) independently records solar and sensor
    azimuth/zenith angles. Since the sun position is identical for all instruments
    at a given time and location, these columns are highly collinear.

    Strategy
    --------
    1. Compute the row-wise mean across all per-band columns for each angle type
       (solar azimuth, sensor azimuth, solar zenith, sensor zenith).
    2. Derive two relative features:
       - ``relative_azimuth``  = |mean_solar_azimuth  − mean_sensor_azimuth|
       - ``relative_zenith``   = |mean_solar_zenith   − mean_sensor_zenith|
    3. Drop all original per-band angle columns to eliminate redundancy.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing per-band solar/sensor angle columns identified by
        substrings ``'solar_azimuth_angle'``, ``'sensor_azimuth_angle'``,
        ``'solar_zenith_angle'``, and ``'sensor_zenith_angle'``.

    Returns
    -------
    pd.DataFrame
        DataFrame with original angle columns removed and two new columns added:
        ``relative_azimuth`` and ``relative_zenith``.

    Example
    -------
    >>> df = relative_mean_angles(df)
    """
    # Map angle type labels to the matching column names in the DataFrame
    angle_map = {
        'solar_azimuth':  [col for col in df.columns if 'solar_azimuth_angle'  in col],
        'sensor_azimuth': [col for col in df.columns if 'sensor_azimuth_angle' in col],
        'solar_zenith':   [col for col in df.columns if 'solar_zenith_angle'   in col],
        'sensor_zenith':  [col for col in df.columns if 'sensor_zenith_angle'  in col],
    }

    # Compute row-wise means for each angle group
    mean_var = pd.DataFrame()
    for name, cols in angle_map.items():
        mean_var[f'mean_{name}'] = df[cols].mean(axis=1)

    # Derive relative angles as the absolute difference between solar and sensor means
    df['relative_azimuth'] = np.abs(mean_var['mean_solar_azimuth'] - mean_var['mean_sensor_azimuth'])
    df['relative_zenith']  = np.abs(mean_var['mean_solar_zenith']  - mean_var['mean_sensor_zenith'])

    # NOTE: Air Mass Factor Proxy via solar zenith (commented out — kept for reference)
    # df['solar_zenith_rad']    = np.radians(df['mean_solar_zenith'])
    # df['air_mass_factor_proxy'] = 1 / np.cos(df['solar_zenith_rad'])

    # Remove all original redundant angle columns
    all_original_angles = [col for list_of_cols in angle_map.values() for col in list_of_cols]
    df.drop(columns=all_original_angles, inplace=True)

    return df


def calculate_air_mass_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Air Mass Factor (AMF) ratios for NO2, SO2, and HCHO.

    The AMF represents the enhancement of the optical path length of light
    through the atmosphere relative to the nadir (vertical) path. Computing it
    helps the model account for geometric and atmospheric effects that influence
    satellite-retrieved pollutant concentration measurements.

    New Columns
    -----------
    - ``AMF_NO2``       : slant NO2 column  / vertical NO2 column
    - ``AMF_SO2_calc``  : slant SO2 column  / vertical SO2 column
    - ``AMF_HCHO_calc`` : slant HCHO column / tropospheric HCHO column

    Infinities produced by division-by-zero are replaced with ``NaN``
    and then filled with ``0``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the required slant and vertical column
        density fields for NO2, SO2, and HCHO.

    Returns
    -------
    pd.DataFrame
        DataFrame with three additional AMF columns appended.

    Example
    -------
    >>> df = calculate_air_mass_factors(df)
    """
    # NO2: slant / vertical column density
    df['AMF_NO2'] = (
        df['L3_NO2_NO2_slant_column_number_density']
        / df['L3_NO2_NO2_column_number_density']
    )

    # SO2: slant / vertical column density
    df['AMF_SO2_calc'] = (
        df['L3_SO2_SO2_slant_column_number_density']
        / df['L3_SO2_SO2_column_number_density']
    )

    # HCHO: slant / tropospheric column density
    df['AMF_HCHO_calc'] = (
        df['L3_HCHO_HCHO_slant_column_number_density']
        / df['L3_HCHO_tropospheric_HCHO_column_number_density']
    )

    # Guard against division-by-zero infinities
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


def calculate_atmospheric_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute physical atmospheric ratio features for NO2 and cloud geometry.

    New Columns
    -----------
    - ``NO2_Tropo_Ratio``          : Fraction of total NO2 column residing in the
                                     troposphere. Values closer to 1.0 indicate
                                     stronger ground-level pollution signal.
    - ``Cloud_Thickness_Pressure`` : Pressure difference between cloud base and
                                     cloud top (hPa). Larger values indicate thicker
                                     clouds that may mask surface-level measurements.

    Infinities produced by division-by-zero are replaced with ``NaN``
    and then filled with ``0``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing tropospheric/total NO2 columns and
        cloud base/top pressure data.

    Returns
    -------
    pd.DataFrame
        DataFrame with two additional atmospheric index columns appended.

    Example
    -------
    >>> df = calculate_atmospheric_indices(df)
    """
    # Tropospheric fraction of total NO2 column (pollution vs. stratospheric background)
    df['NO2_Tropo_Ratio'] = (
        df['L3_NO2_tropospheric_NO2_column_number_density']
        / df['L3_NO2_NO2_column_number_density']
    )

    # Vertical pressure extent of the cloud layer
    df['Cloud_Thickness_Pressure'] = (
        df['L3_CLOUD_cloud_base_pressure']
        - df['L3_CLOUD_cloud_top_pressure']
    )

    # Guard against division-by-zero infinities
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


# ================================================================================
# 3. COLUMN MANAGEMENT
# ================================================================================

def drop_features(df: pd.DataFrame, keywords: list) -> pd.DataFrame:
    """
    Drop all columns whose names contain any of the specified keywords.

    Matching is case-insensitive and substring-based, making this useful
    for bulk-removing entire feature groups (e.g. all CH4 columns, all
    target-leakage columns) without listing exact column names.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to clean.
    keywords : list of str
        Keywords to search for. A column is dropped if *any* keyword
        is found as a substring of its name (case-insensitive).

    Returns
    -------
    pd.DataFrame
        DataFrame with matched columns removed.

    Example
    -------
    >>> df = drop_features(df, keywords=['Place_ID X Date', 'ch4', 'target_'])
    """
    # Identify all columns that match at least one keyword
    to_drop = [
        col for col in df.columns
        if any(key.lower() in col.lower() for key in keywords)
    ]

    df = df.drop(columns=to_drop)
    return df


# ================================================================================
# 4. COLLINEARITY REDUCTION
# ================================================================================

def cloud_fraction_reduction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace per-band cloud fraction columns with a single mean column.

    Multiple satellite instruments each record their own cloud fraction,
    leading to high multicollinearity. This function computes the row-wise
    mean across all ``cloud_fraction`` columns and drops the originals.

    New Column
    ----------
    - ``mean_cloud_fraction`` : Row-wise mean of all ``*cloud_fraction*`` columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing one or more columns with ``'cloud_fraction'``
        in their name.

    Returns
    -------
    pd.DataFrame
        DataFrame with original cloud fraction columns removed and
        ``mean_cloud_fraction`` added.

    Example
    -------
    >>> df = cloud_fraction_reduction(df)
    """
    cloud_frac_cols = [c for c in df.columns if 'cloud_fraction' in c]
    df['mean_cloud_fraction'] = df[cloud_frac_cols].mean(axis=1)
    df = df.drop(columns=cloud_frac_cols)

    return df


def sensor_altitude_reduction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace per-band sensor altitude columns with a single mean column.

    Multiple satellite instruments each record their own sensor altitude,
    leading to high multicollinearity. This function computes the row-wise
    mean across all ``sensor_altitude`` columns and drops the originals.

    New Column
    ----------
    - ``mean_sensor_altitude`` : Row-wise mean of all ``*sensor_altitude*`` columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing one or more columns with ``'sensor_altitude'``
        in their name.

    Returns
    -------
    pd.DataFrame
        DataFrame with original sensor altitude columns removed and
        ``mean_sensor_altitude`` added.

    Example
    -------
    >>> df = sensor_altitude_reduction(df)
    """
    sensor_altitude_cols = [c for c in df.columns if 'sensor_altitude' in c]
    df['mean_sensor_altitude'] = df[sensor_altitude_cols].mean(axis=1)
    df = df.drop(columns=sensor_altitude_cols)

    return df


# ================================================================================
# 5. IMPUTATION
# ================================================================================

def impute_full_pipeline(
    df: pd.DataFrame,
    exclude_cols: list = ['Place_ID', 'Date', 'target'],
    fit_stats: dict = None
) -> tuple:
    """
    Impute missing numeric values using a grouped-mean strategy with a global fallback.

    Designed to be called twice — once on training data (to fit and transform),
    and once on validation/test data (to transform only using the fitted stats).
    This prevents data leakage from the validation/test sets.

    Strategy
    --------
    1. Replace ``0`` with ``NaN`` (sensor zeros treated as missing).
    2. For each numeric column, compute the per-``Place_ID`` mean from training data.
    3. Fill ``NaN`` values using the station-level mean.
    4. Fall back to the global median for any remaining ``NaN`` (unknown stations
       or fully empty groups).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to impute. Must contain a ``'Place_ID'`` column.
    exclude_cols : list of str, optional
        Columns to skip during imputation. Defaults to
        ``['Place_ID', 'Date', 'target']``.
    fit_stats : dict or None, optional
        - ``None``  → fit mode (training): stats are computed from ``df``.
        - ``dict``  → transform mode (val/test): pre-computed stats are applied.

    Returns
    -------
    df_copy : pd.DataFrame
        Imputed copy of the input DataFrame.
    fit_stats : dict
        Dictionary of per-column imputation statistics (lookup table + global median).
        Pass this into subsequent calls for validation/test sets.

    Example
    -------
    >>> # Training set — fit and transform
    >>> df_train_imputed, fit_stats = impute_full_pipeline(df_train)

    >>> # Validation set — transform only using training stats
    >>> df_val_imputed, _ = impute_full_pipeline(df_val, fit_stats=fit_stats)
    """
    df_copy = df.copy()

    # Select only numeric columns not in the exclusion list
    cols_to_fix = [
        c for c in df_copy.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    # Determine whether we are fitting (train) or transforming (val/test)
    if fit_stats is None:
        fit_stats = {}
        is_train = True
    else:
        is_train = False

    for col in cols_to_fix:
        # Treat sensor zeros as missing
        df_copy[col] = df_copy[col].replace(0, np.nan)

        if is_train:
            # FIT: Learn imputation statistics from training data only
            global_med   = df_copy[col].median()
            group_lookup = df_copy.groupby('Place_ID')[col].mean().fillna(global_med)
            fit_stats[col] = {'lookup': group_lookup, 'median': global_med}
        else:
            # TRANSFORM: Apply pre-computed statistics from training
            group_lookup = fit_stats[col]['lookup']
            global_med   = fit_stats[col]['median']

        # Step 1: Fill using station-level mean
        df_copy[col] = df_copy[col].fillna(df_copy['Place_ID'].map(group_lookup))

        # Step 2: Global median fallback for unknown stations or fully empty groups
        df_copy[col] = df_copy[col].fillna(global_med)

    return df_copy, fit_stats


class GroupedMedianImputer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible imputer using per-group medians with a global median fallback.

    Designed to slot into a ``sklearn.pipeline.Pipeline``. Fits group-level
    median statistics on the training set and applies them during transform,
    preventing data leakage.

    Strategy
    --------
    1. ``fit``      : Compute per-``Place_ID`` median and global median for each
                      numeric column from ``X_train``.
    2. ``transform``: Fill ``NaN`` values using the station median; fall back to
                      the global median for unknown stations or fully empty groups.

    Attributes
    ----------
    group_medians : pd.DataFrame
        Per-``Place_ID`` median values learned during ``fit``.
    global_medians : pd.Series
        Global median values learned during ``fit``.

    Example
    -------
    >>> from sklearn.pipeline import Pipeline
    >>> pipe = Pipeline([
    ...     ('imputer', GroupedMedianImputer()),
    ...     ('model',   SomeModel())
    ... ])
    >>> pipe.fit(X_train, y_train)
    >>> pipe.predict(X_val)
    """

    def __init__(self):
        self.group_medians  = None
        self.global_medians = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Learn per-group and global median statistics from training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data. Must contain a ``'Place_ID'`` column.
        y : ignored

        Returns
        -------
        self
        """
        self.group_medians  = X.groupby('Place_ID').median(numeric_only=True)
        self.global_medians = X.median(numeric_only=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using fitted statistics.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform. Must contain a ``'Place_ID'`` column.

        Returns
        -------
        pd.DataFrame
            Imputed copy of ``X``.
        """
        X = X.copy()
        for col in self.group_medians.columns:
            # Primary fill: station-level median
            X[col] = X[col].fillna(X['Place_ID'].map(self.group_medians[col]))
            # Fallback: global median for unseen stations or fully null groups
            X[col] = X[col].fillna(self.global_medians[col])
        return X

    def set_output(self, transform=None):
        """
        Enable compatibility with ``pipeline.set_output(transform='pandas')``.

        Parameters
        ----------
        transform : str or None
            Output container type (e.g. ``'pandas'``). Accepted but not used.

        Returns
        -------
        self
        """
        return self


# ================================================================================
# 6. RESULT PLOTS
# ================================================================================

def plot_model_comparison(results: dict, save_path: str = None):
    """
    Plot a grouped bar chart comparing RMSE and R² across models and splits.

    Renders two side-by-side panels — one for RMSE (lower is better) and one
    for R² (higher is better) — with Train, Validation, and Test bars grouped
    per model. Each bar is annotated with its value.

    Parameters
    ----------
    results : dict
        Nested dictionary of the form::

            {
                'ModelName': {
                    'Train':      {'RMSE': float, 'MAE': float, 'R2': float},
                    'Validation': {'RMSE': float, 'MAE': float, 'R2': float},
                    'Test':       {'RMSE': float, 'MAE': float, 'R2': float},
                },
                ...
            }

        This is the ``results`` dict populated by the model comparison loop
        in the pipeline notebook.
    save_path : str or None, optional
        Full file path to save the figure (e.g. ``'Images/model_comparison.png'``).
        If ``None``, the figure is only displayed and not saved. Defaults to ``None``.

    Returns
    -------
    None
        Displays the chart inline via ``plt.show()``. Optionally saves to disk.

    Example
    -------
    >>> plot_model_comparison(results)
    >>> plot_model_comparison(results, save_path='Images/model_comparison.png')
    """
    model_names = list(results.keys())
    splits      = ['Train', 'Validation', 'Test']
    colours     = ['#4C91C9', '#F5A623', '#5DBB7A']

    rmse_values = {s: [results[m][s]['RMSE'] for m in model_names] for s in splits}
    r2_values   = {s: [results[m][s]['R2']   for m in model_names] for s in splits}

    x     = np.arange(len(model_names))
    width = 0.26

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#F8F9FA')

    for ax in (ax1, ax2):
        ax.set_facecolor('#F8F9FA')
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#CCCCCC')
        ax.yaxis.grid(True, color='#E0E0E0', linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    # RMSE panel
    for i, (split, colour) in enumerate(zip(splits, colours)):
        bars = ax1.bar(x + (i - 1) * width, rmse_values[split], width,
                       label=split, color=colour, alpha=0.88, edgecolor='white', zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f'{h:.1f}',
                     ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#444')

    ax1.set_title('RMSE by Model & Split\n(lower is better)', fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylabel('RMSE', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=11, fontweight='bold')
    ax1.set_ylim(0, max(max(v) for v in rmse_values.values()) * 1.25)
    ax1.legend(framealpha=0, fontsize=9)

    # R² panel
    for i, (split, colour) in enumerate(zip(splits, colours)):
        bars = ax2.bar(x + (i - 1) * width, r2_values[split], width,
                       label=split, color=colour, alpha=0.88, edgecolor='white', zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f'{h:.3f}',
                     ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#444')

    ax2.set_title('R² Score by Model & Split\n(higher is better)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylabel('R²', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, fontsize=11, fontweight='bold')
    ax2.set_ylim(0, min(max(max(v) for v in r2_values.values()) * 1.25, 1.0))
    ax2.legend(framealpha=0, fontsize=9)

    fig.suptitle('Model Performance Comparison — Air Quality Prediction',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred, model_name: str = 'Model', metrics: dict = None, save_path: str = None):
    """
    Scatter plot of actual vs predicted values with a perfect-prediction diagonal.

    Points along the red dashed line represent perfect predictions. Scatter
    around it shows the model's uncertainty. An optional metrics annotation
    box can be displayed in the top-left corner.

    Parameters
    ----------
    y_true : array-like
        Ground truth target values (e.g. ``y_test``).
    y_pred : array-like
        Predicted target values from the model.
    model_name : str, optional
        Model label used in the chart title. Defaults to ``'Model'``.
    metrics : dict or None, optional
        Optional dictionary of metrics to annotate on the chart, e.g.::

            {'RMSE': 29.306, 'MAE': 19.262, 'R2': 0.610}

        If ``None``, no annotation box is shown.
    save_path : str or None, optional
        Full file path to save the figure (e.g. ``'Images/actual_vs_predicted.png'``).
        If ``None``, the figure is only displayed and not saved. Defaults to ``None``.

    Returns
    -------
    None
        Displays the chart inline via ``plt.show()``. Optionally saves to disk.

    Example
    -------
    >>> plot_actual_vs_predicted(
    ...     y_test, y_pred_test,
    ...     model_name='Stacking',
    ...     metrics={'RMSE': 29.306, 'MAE': 19.262, 'R2': 0.610},
    ...     save_path='Images/actual_vs_predicted.png'
    ... )
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#CCCCCC')

    ax.scatter(y_true, y_pred, alpha=0.35, s=18,
               color='#4C91C9', edgecolors='none', label='Test predictions')

    # Perfect prediction diagonal
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val],
            color='#E05C5C', linewidth=1.8, linestyle='--', label='Perfect prediction')

    ax.set_xlabel('Actual Values', fontsize=11)
    ax.set_ylabel('Predicted Values', fontsize=11)
    ax.set_title(f'Actual vs Predicted — {model_name} (Test Set)',
                 fontsize=12, fontweight='bold', pad=12)
    ax.legend(framealpha=0, fontsize=9)

    # Optional metrics annotation box
    if metrics is not None:
        annotation = '\n'.join(f'{k} = {v}' for k, v in metrics.items())
        ax.text(0.05, 0.85, annotation,
                transform=ax.transAxes, fontsize=9, color='#333333',
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          alpha=0.7, edgecolor='#CCCCCC'))

    plt.tight_layout()
    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_residuals(y_true, y_pred, model_name: str = 'Model', save_path: str = None):
    """
    Plot residuals vs predicted values and a residual distribution histogram.

    Two panels side by side:
    - **Left** — Residuals (Actual − Predicted) vs Predicted values. A well-
      behaved model shows residuals scattered evenly around zero with no
      clear pattern (no heteroscedasticity or bias).
    - **Right** — Histogram of residuals with zero-error and mean-residual
      lines. Ideally the distribution should be approximately normal and
      centred near zero.

    Parameters
    ----------
    y_true : array-like
        Ground truth target values (e.g. ``y_test``).
    y_pred : array-like
        Predicted target values from the model.
    model_name : str, optional
        Model label used in chart titles. Defaults to ``'Model'``.
    save_path : str or None, optional
        Full file path to save the figure (e.g. ``'Images/residuals.png'``).
        If ``None``, the figure is only displayed and not saved. Defaults to ``None``.

    Returns
    -------
    None
        Displays the chart inline via ``plt.show()``. Optionally saves to disk.

    Example
    -------
    >>> plot_residuals(y_test, y_pred_test, model_name='Stacking')
    >>> plot_residuals(y_test, y_pred_test, model_name='Stacking', save_path='Images/residuals.png')
    """
    residuals = np.array(y_true) - np.array(y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#F8F9FA')

    for ax in (ax1, ax2):
        ax.set_facecolor('#F8F9FA')
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#CCCCCC')
        ax.yaxis.grid(True, color='#E0E0E0', linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    # Left panel — residuals vs predicted
    ax1.scatter(y_pred, residuals, alpha=0.35, s=18,
                color='#4C91C9', edgecolors='none')
    ax1.axhline(0, color='#E05C5C', linewidth=1.8, linestyle='--', label='Zero error')
    ax1.set_xlabel('Predicted Values', fontsize=11)
    ax1.set_ylabel('Residuals (Actual − Predicted)', fontsize=11)
    ax1.set_title(f'Residuals vs Predicted\n{model_name} (Test Set)',
                  fontsize=12, fontweight='bold', pad=10)
    ax1.legend(framealpha=0, fontsize=9)

    # Right panel — residual histogram
    ax2.hist(residuals, bins=45, color='#4C91C9', alpha=0.80,
             edgecolor='white', linewidth=0.5, zorder=3)
    ax2.axvline(0, color='#E05C5C', linewidth=1.8,
                linestyle='--', label='Zero error')
    ax2.axvline(residuals.mean(), color='#F5A623', linewidth=1.5,
                linestyle='-', label=f'Mean residual ({residuals.mean():.2f})')
    ax2.set_xlabel('Residual Value', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title(f'Residual Distribution\n{model_name} (Test Set)',
                  fontsize=12, fontweight='bold', pad=10)
    ax2.legend(framealpha=0, fontsize=9)

    plt.tight_layout()
    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
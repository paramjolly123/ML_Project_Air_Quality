import pandas as pd
import numpy as np


## --------------------------- Correlation heatmap for Target -------------------------------------------------

def plot_target_correlation(data: pd.DataFrame, target_col: list, title="Feature Correlation with Target"):
    """
    Plots a sorted heatmap showing the correlation of all features with the target.

    Args -->    data (dataframe), 
                'feature_name' (feature name with ' ')

    Returns --> signle column heatmap plot for target vs feature correlation
    """
    #1. Correlation for numerical data only
    correlations = data.corr(numeric_only=True)[[target_col]]
    correlations = correlations.sort_values(by=target_col, ascending=False)
    correlations = correlations.drop(target_col)

    plt.figure(figsize=(5, 8))
    sns.heatmap(correlations, annot=True, cmap='viridis',
                vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
    plt.title(f"{title}: {target_col}")
    plt.show()



## ---------------------- Relative Mean Calculation (for 'azimuth' and 'zenith') Function ----------------------


def relative_mean_angles(df: pd.DataFrame):
    """
    Aggregates redundant satellite observation angles and computes relative geometric features.
    Calculating raw wise mean for each pollutants --> seperatley for 'azimuth' and 'zenith'
    calculating the raltive 'azimuth' and 'zenith'
    dropping the original features

    Args --> df (pd.DataFrame): dataframe where to do Relative Mean Calculation for 'azimuth' and 'zenith'

    Returns --> dataframe
    """
    # 1. Map columns into their respective geometric categories
    angle_map = {
        'solar_azimuth': [col for col in df.columns if 'solar_azimuth_angle' in col],
        'sensor_azimuth': [col for col in df.columns if 'sensor_azimuth_angle' in col],
        'solar_zenith': [col for col in df.columns if 'solar_zenith_angle' in col],
        'sensor_zenith': [col for col in df.columns if 'sensor_zenith_angle' in col]
    }

    # 2. Calculate the mean for each category (Dimensionality Reduction)
    mean_var = pd.DataFrame()
    for name, cols in angle_map.items():
        mean_var[f'mean_{name}'] = df[cols].mean(axis=1)

    # 3. Compute Relative Azimuth (Horizontal Geometry)
    df['relative_azimuth'] = np.abs(mean_var['mean_solar_azimuth'] - mean_var['mean_sensor_azimuth'])

    # 4. Compute Relative Zenith (Vertical Geometry / Path Length)
    df['relative_zenith'] = np.abs(mean_var['mean_solar_zenith'] - mean_var['mean_sensor_zenith'])

    # 5. Clean up: Remove all the original redundant angle columns
    all_original_angles = [col for list_of_cols in angle_map.values() for col in list_of_cols]
    df.drop(columns=all_original_angles, inplace=True)

    print(f"Reduced features. New columns added: ['relative_azimuth', 'relative_zenith']")
    return df


## --------------------------- Drop Feature Function -------------------------------

def drop_features(df: pd.DataFrame, keywords: list):
    """
    Removes columns from the DataFrame that contain any of the specified keywords (case-insensitive).
    Identifies matches by checking if any keyword is a substring of the column name.

    Arg -->     df (pd.DataFrame): 
                keyword: a list of features to remove from dataframe
    
    Returns --> cleaned DataFrame and prints the number of columns removed.
    """
    
    # Create a list of columns that match any keyword
    to_drop = [
        col for col in df.columns 
        if any(key.lower() in col.lower() for key in keywords)
    ]
    
    # Drop them all in one go
    df_cleaned = df.drop(columns=to_drop)
    
    print(f"Dropped {len(to_drop)} columns. New shape: {df_cleaned.shape}")
    return df_cleaned

# Usage:
#drop_features = ['Place_ID X Date', 'ch4', 'target_']
#df = clean_columns(df, drop_features)


## --------------------------- Air Mass Factor Calculation Function -------------------------------

def calculate_air_mass_factors(df: pd.DataFrame):
    """
    Calculates the Air Mass Factor (AMF) as a ratio of slant to vertical column densities.

    The AMF represents the enhancement of the optical path length of light through 
    the atmosphere; calculating it helps the model account for geometric and 
    atmospheric effects on pollutant concentration measurements.

    Args -->    df (pd.DataFrame): Input DataFrame containing slant and vertical column densities for NO2, SO2, and HCHO.

    Returns -->  pd.DataFrame: DataFrame with three new columns: 'AMF_NO2', 'AMF_SO2_calc', and 'AMF_HCHO_calc'.
    """
    # NO2 AMF
    df['AMF_NO2'] = df['L3_NO2_NO2_slant_column_number_density'] / df['L3_NO2_NO2_column_number_density']
    
    # SO2 AMF (Using the slant and column density)
    df['AMF_SO2_calc'] = df['L3_SO2_SO2_slant_column_number_density'] / df['L3_SO2_SO2_column_number_density']
    
    # HCHO AMF (Using slant and tropospheric column density)
    df['AMF_HCHO_calc'] = df['L3_HCHO_HCHO_slant_column_number_density'] / df['L3_HCHO_tropospheric_HCHO_column_number_density']
    
    return df

## --------------------------- Atmospheric Indices Calculation Function -------------------------------


def calculate_atmospheric_indices(df: pd.DataFrame):
    """
    Computes environmental ratios and physical cloud metrics for atmospheric analysis.

    Calculates the NO2 Tropospheric Ratio to isolate ground-level pollution from 
    stratospheric interference and computes Cloud Pressure Thickness to quantify 
    the vertical extent and potential masking effect of cloud cover.

    Args -->       df (pd.DataFrame): Input DataFrame containing tropospheric/total NO2 columns and cloud base/top pressure data.

    Returns -->    pd.DataFrame: DataFrame with two new columns: 'NO2_Tropo_Ratio' and 'Cloud_Thickness_Pressure'.
    """
    # NO2 Tropo Ratio: How much of the total NO2 is in the troposphere?
    df['NO2_Tropo_Ratio'] = df['L3_NO2_tropospheric_NO2_column_number_density'] / df['L3_NO2_NO2_column_number_density']
    
    # Cloud pressure thickness (Base - Top)
    df['Cloud_Thickness_Pressure'] = df['L3_CLOUD_cloud_base_pressure'] - df['L3_CLOUD_cloud_top_pressure']
    
    return df
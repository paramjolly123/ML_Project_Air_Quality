# ML_Project_Air_Quality


### **Detailed Feature Band Description (74 Total)**

| Band Name | Units | Min | Max | Pixel | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **METEOROLOGY (NOAA GFS)** | | | | | **Weather Context** |
| `precipitable_water_entire_atmosphere` | kg/m² | 0.42 | 72.60 | 1113m | Total water vapor in air column |
| `relative_humidity_2m_above_ground` | % | 5.13 | 100.00 | 1113m | Relative humidity at 2m height |
| `specific_humidity_2m_above_ground` | kg/kg | 0.00 | 0.02 | 1113m | Mass of water vapor per kg of air |
| `temperature_2m_above_ground` | K | -34.65 | 37.44 | 1113m | Air temperature at 2m (Kelvin) |
| `u_component_of_wind_10m_above_ground` | m/s | -15.56 | 17.96 | 1113m | Eastward wind component |
| `v_component_of_wind_10m_above_ground` | m/s | -15.02 | 15.06 | 1113m | Northward wind component |
| **NITROGEN DIOXIDE (L3_NO2)** | | | | | **Primary Pollutant** |
| `L3_NO2_NO2_column_number_density` | mol/m² | -0.00 | 0.0030 | 1113m | Total vertical column of NO2 |
| `L3_NO2_NO2_slant_column_number_density`| mol/m² | 0.00 | 0.0024 | 1113m | NO2 density along the line of sight |
| `L3_NO2_absorbing_aerosol_index` | Unitless | -3.94 | 4.82 | 1113m | UV-absorbing aerosol index |
| `L3_NO2_cloud_fraction` | 0-1 | 0.00 | 1.00 | 1113m | Fraction of pixel covered by clouds |
| `L3_NO2_sensor_altitude` | m | 0.00 | 844494 | 1113m | Altitude of the satellite |
| `L3_NO2_sensor_azimuth_angle` | Degrees | -111.47 | 79.16 | 1113m | Azimuth angle of the satellite |
| `L3_NO2_sensor_zenith_angle` | Degrees | 0.00 | 66.25 | 1113m | Zenith angle of the satellite |
| `L3_NO2_solar_azimuth_angle` | Degrees | -179.95 | 179.75 | 1113m | Azimuth angle of the Sun |
| `L3_NO2_solar_zenith_angle` | Degrees | 0.00 | 81.19 | 1113m | Zenith angle of the Sun |
| `L3_NO2_stratospheric_NO2...` | mol/m² | 0.00 | 0.0001 | 1113m | NO2 in the stratosphere |
| `L3_NO2_tropopause_pressure` | Pa | 0.00 | 31592 | 1113m | Pressure at the tropopause |
| `L3_NO2_tropospheric_NO2...` | mol/m² | -0.00 | 0.0021 | 1113m | NO2 in the troposphere (lower air) |
| **CARBON MONOXIDE (L3_CO)** | | | | | **Primary Pollutant** |
| `L3_CO_CO_column_number_density` | mol/m² | 0.00 | 0.29 | 1113m | Total vertical column of CO |
| `L3_CO_H2O_column_number_density` | mol/m² | 0.00 | 19656 | 1113m | Vertical column of water vapor |
| `L3_CO_cloud_height` | m | -489.87 | 4999.3 | 1113m | Height of clouds for CO correction |
| `L3_CO_sensor_altitude` | m | 0.00 | 844553 | 1113m | Altitude of the satellite |
| `L3_CO_sensor_azimuth_angle` | Degrees | -106.06 | 78.14 | 1113m | Azimuth angle of the satellite |
| `L3_CO_sensor_zenith_angle` | Degrees | 0.00 | 65.42 | 1113m | Zenith angle of the satellite |
| `L3_CO_solar_azimuth_angle` | Degrees | -179.88 | 179.98 | 1113m | Azimuth angle of the Sun |
| `L3_CO_solar_zenith_angle` | Degrees | 0.00 | 79.99 | 1113m | Zenith angle of the Sun |
| **OZONE (L3_O3)** | | | | | **Secondary Pollutant** |
| `L3_O3_O3_column_number_density` | mol/m² | 0.00 | 0.24 | 1113m | Total vertical column of Ozone |
| `L3_O3_O3_effective_temperature` | K | 0.00 | 238.87 | 1113m | Effective temperature of Ozone layer |
| `L3_O3_cloud_fraction` | 0-1 | 0.00 | 1.00 | 1113m | Cloud fraction during O3 reading |
| `L3_O3_sensor_azimuth_angle` | Degrees | -111.95 | 77.71 | 1113m | Observation angle (Azimuth) |
| `L3_O3_sensor_zenith_angle` | Degrees | 0.00 | 66.06 | 1113m | Observation angle (Zenith) |
| `L3_O3_solar_azimuth_angle` | Degrees | -179.83 | 179.59 | 1113m | Solar position (Azimuth) |
| `L3_O3_solar_zenith_angle` | Degrees | 0.00 | 79.65 | 1113m | Solar position (Zenith) |
| **SULPHUR DIOXIDE (L3_SO2)** | | | | | **Primary Pollutant** |
| `L3_SO2_SO2_column_number_density` | mol/m² | -0.04 | 0.03 | 1113m | Total vertical column of SO2 |
| `L3_SO2_SO2_column_number_density_amf`| Unitless | 0.00 | 2.18 | 1113m | Air Mass Factor for SO2 |
| `L3_SO2_SO2_slant_column_number_density`| mol/m² | -0.004 | 0.007 | 1113m | SO2 density along line of sight |
| `L3_SO2_absorbing_aerosol_index` | Unitless | -4.83 | 3.74 | 1113m | UV-absorbing aerosol index |
| `L3_SO2_cloud_fraction` | 0-1 | 0.00 | 0.62 | 1113m | Cloud fraction during SO2 reading |
| `L3_SO2_sensor_azimuth_angle` | Degrees | -112.00 | 80.54 | 1113m | Sensor Azimuth |
| `L3_SO2_sensor_zenith_angle` | Degrees | 0.00 | 66.11 | 1113m | Sensor Zenith |
| `L3_SO2_solar_azimuth_angle` | Degrees | -179.88 | 179.78 | 1113m | Solar Azimuth |
| `L3_SO2_solar_zenith_angle` | Degrees | 0.00 | 79.63 | 1113m | Solar Zenith |
| **FORMALDEHYDE (L3_HCHO)** | | | | | **Volatile Organic Comp.** |
| `L3_HCHO_HCHO_slant_column...` | mol/m² | -0.0006| 0.0006 | 1113m | Formaldehyde slant column density |
| `L3_HCHO_cloud_fraction` | 0-1 | 0.00 | 0.62 | 1113m | Cloud fraction for HCHO |
| `L3_HCHO_sensor_azimuth_angle` | Degrees | -112.16 | 81.78 | 1113m | Sensor Azimuth |
| `L3_HCHO_sensor_zenith_angle` | Degrees | 0.00 | 66.13 | 1113m | Sensor Zenith |
| `L3_HCHO_solar_azimuth_angle` | Degrees | -179.83 | 179.80 | 1113m | Solar Azimuth |
| `L3_HCHO_solar_zenith_angle` | Degrees | 0.00 | 79.64 | 1113m | Solar Zenith |
| `L3_HCHO_tropospheric_HCHO...` | mol/m² | -0.0006| 0.0010 | 1113m | Formaldehyde in lower atmosphere |
| `L3_HCHO_tropospheric_HCHO_amf` | Unitless | 0.00 | 3.10 | 1113m | Air Mass Factor for HCHO |
| **AEROSOL INDEX (L3_AER_AI)** | | | | | **Particulate Indicator** |
| `L3_AER_AI_absorbing_aerosol_index` | Unitless | -3.97 | 4.82 | 1113m | Main aerosol index measurement |
| `L3_AER_AI_sensor_altitude` | m | 828758 | 844494 | 1113m | Altitude of satellite |
| `L3_AER_AI_sensor_azimuth_angle` | Degrees | -112.09 | 77.71 | 1113m | Sensor Azimuth |
| `L3_AER_AI_sensor_zenith_angle` | Degrees | 0.42 | 66.50 | 1113m | Sensor Zenith |
| `L3_AER_AI_solar_azimuth_angle` | Degrees | -179.46 | 179.59 | 1113m | Solar Azimuth |
| `L3_AER_AI_solar_zenith_angle` | Degrees | 8.25 | 87.49 | 1113m | Solar Zenith |
| **CLOUD (L3_CLOUD)** | | | | | **Interference Correction** |
| `L3_CLOUD_cloud_base_height` | m | 9.00 | 14000 | 1113m | Height of cloud base |
| `L3_CLOUD_cloud_base_pressure` | Pa | 12936 | 101299 | 1113m | Pressure at cloud base |
| `L3_CLOUD_cloud_fraction` | 0-1 | 0.00 | 1.00 | 1113m | Effective cloud fraction |
| `L3_CLOUD_cloud_optical_depth` | Unitless | 1.00 | 237.12 | 1113m | Cloud optical thickness |
| `L3_CLOUD_cloud_top_height` | m | 10.05 | 15000 | 1113m | Height of cloud top |
| `L3_CLOUD_cloud_top_pressure` | Pa | 10957 | 101299 | 1113m | Pressure at cloud top |
| `L3_CLOUD_sensor_azimuth_angle` | Degrees | -111.95 | 77.71 | 1113m | Sensor Azimuth |
| `L3_CLOUD_sensor_zenith_angle` | Degrees | 0.40 | 66.06 | 1113m | Sensor Zenith |
| `L3_CLOUD_solar_azimuth_angle` | Degrees | -179.83 | 179.59 | 1113m | Solar Azimuth |
| `L3_CLOUD_solar_zenith_angle` | Degrees | 8.40 | 79.65 | 1113m | Solar Zenith |
| `L3_CLOUD_surface_albedo` | 0-1 | 0.02 | 0.99 | 1113m | Earth's surface reflectivity |
| **METHANE (L3_CH4)** | | | | | **Greenhouse Gas** |
| `L3_CH4_CH4_column_volume_mixing...`| ppb | 0.00 | 2112.5 | 1113m | Methane concentration |
| `L3_CH4_aerosol_height` | m | 0.00 | 6478.6 | 1113m | Mean aerosol layer height |
| `L3_CH4_aerosol_optical_depth` | Unitless | 0.00 | 0.21 | 1113m | Optical thickness of aerosols |
| `L3_CH4_sensor_azimuth_angle` | Degrees | -105.37 | 77.36 | 1113m | Sensor Azimuth |
| `L3_CH4_sensor_zenith_angle` | Degrees | 0.00 | 59.97 | 1113m | Sensor Zenith |
| `L3_CH4_solar_azimuth_angle` | Degrees | -179.95 | 179.81 | 1113m | Solar Azimuth |
| `L3_CH4_solar_zenith_angle` | Degrees | 0.00 | 69.99 | 1113m | Solar Zenith |
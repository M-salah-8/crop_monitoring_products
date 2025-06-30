from os.path import basename

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pandas import DataFrame
from geopandas import GeoDataFrame


def fill_missing(daily_df: DataFrame, data_name: str) -> None:
    daily_df.at[0,data_name] = daily_df.at[daily_df[data_name].first_valid_index(), data_name]
    # daily_df.at[0,data_name] = 0
    daily_df[data_name] = daily_df[data_name].interpolate('linear')
    # daily_df[data_name] = daily_df[data_name].interpolate(method='polynomial', order=3)
    # daily_df.fillna(method='ffill')


def write_df_field(
    field_gdf: GeoDataFrame,
    name_attribute: str,
    df: DataFrame,
    tifs_dir: str,
    data_name: str,
) -> DataFrame:
    with rasterio.open(tifs_dir[0]) as src:
        field_gdf = field_gdf.to_crs(src.crs)
    for _, row in field_gdf.iterrows():
        field = row[name_attribute]
        for tif_dir in tifs_dir:
            date = basename(tif_dir).split(".")[0].split("_")[1]
            with rasterio.open(tif_dir) as src:
                array = rasterio.mask.mask(
                    src, [row.geometry], crop=True, nodata=np.nan
                )[0][0]
                df.loc[df["date"] == date, f"{data_name}_{field}"] = np.nanmean(array)
    return df


def write_df_project(
    project_gdf: GeoDataFrame, df: DataFrame, tifs_dir: str, data_name: str
) -> DataFrame:
    with rasterio.open(tifs_dir[0]) as src:
        project_gdf = project_gdf.to_crs(src.crs)
    for tif_dir in tifs_dir:
        date = basename(tif_dir).split(".")[0].split("_")[1]
        with rasterio.open(tif_dir) as src:
            array = src.read(1, window=from_bounds(*project_gdf.total_bounds, transform=src.transform))
        df.loc[df['date'] == date, f"{data_name}"] = np.nanmean(array)
    return df


def fill_et_df(df) -> None:
    # get all columns that starts with kc
    kc_columns = [column for column in df.columns if column.startswith('kc')]
    for kc_column in kc_columns:
        fill_missing(df, kc_column)
        eta_column = kc_column.replace('kc', 'eta')
        etp_column = kc_column.replace('kc', 'etp')
        df[eta_column] = df[etp_column] * df[kc_column]

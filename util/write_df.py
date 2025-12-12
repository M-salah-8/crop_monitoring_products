from os.path import basename, join, exists
from glob import glob
from datetime import datetime

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pandas import DataFrame, date_range, DatetimeIndex
from geopandas import GeoDataFrame, read_file


def fill_missing(dfs: dict[str, DataFrame], ids: list) -> None:
    for df_name, df in dfs.items():
        if df_name == "eta":
            continue
        for id in ids:
            id = str(id)
            df.at[0,id] = df.at[df[id].first_valid_index(), id]
            # df.at[0,id] = 0
            df[id] = df[id].interpolate('linear')
            # df[id] = df[id].interpolate(method='polynomial', order=3)
            # df.fillna(method='ffill')
    if "eta" in dfs and "kc" in dfs and "etp" in dfs:
        for id in ids:
            id = str(id)
            dfs["eta"][id] = dfs["kc"][id] * dfs["etp"][id]


def creat_date_df(initial_date: str, final_date: str) -> DatetimeIndex:
    try:    # TODO: validate first date not later than final date
        initial_datetime = datetime.strptime(initial_date, "%Y-%m-%d")
        final_datetime = datetime.strptime(final_date, "%Y-%m-%d")
    except ValueError as e:
        print(str(e))
        exit()

    # convert to strings
    daily_dates = date_range(
        start=initial_datetime, end=final_datetime, freq="D"
    )
    return daily_dates


def fill_df(product_dir: str, dates: DatetimeIndex, ids: list[int]) -> DataFrame:
    template_gpkg = next((gpkg for gpkg in glob(join(product_dir, "*.gpkg"))), None)
    assert template_gpkg is not None
    gdf = read_file(template_gpkg)
    product_ids: list[int] = gdf["id"].values.tolist()
    assert product_ids == ids
    df = DataFrame({"date": dates})
    for id in ids:
        df[str(id)] = np.nan
    for date in dates:
        gpkg = join(product_dir, f"{date.strftime("%Y-%m-%d")}.gpkg")
        if exists(gpkg):
            gdf = read_file(gpkg)
            for id in ids:
                shape_value: float = gdf.loc[gdf["id"] == id, "stats_median"].values[0]
                df.loc[df["date"] == date, str(id)] = shape_value

    return df


def kc_df(dfs: dict[str, DataFrame], ids: list) -> None:
    dfs["kc"] = dfs["eta"].copy()
    for id in ids:
        id = str(id)
        dfs["kc"][id] = dfs["eta"][id] / dfs["etp"][id]


def products_dfs(
    initial_date: str,
    final_date: str,
    products: dict[str, str],
) -> dict[str, DataFrame]:
    dfs: dict[str, DataFrame] = {}
    first_product = next(iter(products.values()), "")
    template_gpkg = next((gpkg for gpkg in glob(join(first_product, "*.gpkg"))), None)
    assert template_gpkg is not None
    gdf = read_file(template_gpkg)
    ids: list[int] = gdf["id"].values.tolist()
    dates = creat_date_df(initial_date, final_date)
    for product_name, product_dir in products.items():
        df = fill_df(product_dir, dates, ids)
        dfs[product_name] = df
    if "eta" in dfs and "etp" in dfs:
        kc_df(dfs, ids)

    fill_missing(dfs, ids)

    return dfs

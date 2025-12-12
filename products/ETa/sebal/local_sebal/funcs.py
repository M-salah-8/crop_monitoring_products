from os.path import join
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray
import rasterio
import pyproj


REQUIRED_METADATA = [
    "SPACECRAFT_ID",
    "SCENE_CENTER_TIME",
    "SUN_ELEVATION",
    "utc_timestamp",
    "DATE_ACQUIRED",
]


def save_data(output_dir: str, meta: dict, array: NDArray, name: str) -> str:
    tif_file = join(output_dir, f"{name.lower()}.tif")
    with rasterio.open(tif_file, "w", **meta) as dst:
        dst.write(array, 1)
    return tif_file


def save_to_image(
    image: dict[str, str], output_dir: str, meta: dict, array: NDArray, name: str
) -> None:
    image[name.upper()] = save_data(output_dir, meta, array, name)


def get_image_tif_metadata(tif: str) -> tuple[dict, int, tuple[float, float], list[float]]:
    with rasterio.open(tif) as src:
        tif_meta = src.meta
        res = src.res
        bounds = src.bounds
    original_nodata: int = tif_meta["nodata"]
    tif_meta.update(dtype= np.float32, nodata= np.nan)
    return tif_meta, original_nodata, res, bounds


def open_band(band_name: str, image_files: list[str], image_dir: str) -> NDArray:
    band_check = [
        band_file for band_file in image_files if band_file.endswith(f"{band_name}.TIF")
    ]
    assert len(band_check) == 1
    tif = join(image_dir, band_check[0])
    with rasterio.open(tif) as src:
        array = src.read(1)

    return array


def _open_band(
    band_args=tuple[
        str,
        float,
        float,
        NDArray[np.bool_],
        list[str],
        str,
    ],
) -> NDArray:
    band_name, scale, offset, image_mask, image_files, image_dir = band_args

    band_check = [
        band_file for band_file in image_files if band_file.endswith(f"{band_name}.TIF")
    ]
    assert len(band_check) == 1
    tif = join(image_dir, band_check[0])
    with rasterio.open(tif) as src:
        array = src.read(1).astype(np.float32)
    array[image_mask] = np.nan
    return array * scale + offset


def open_bands(
    bands: dict, image_mask: NDArray[np.bool_], image_files: list[str], image_dir: str
) -> list[NDArray]:
    scale: float = bands["scale"]
    offset: float = bands["offset"]
    with ThreadPoolExecutor() as executor:
        arrays = list(
            executor.map(
                _open_band,
                [
                    (band, scale, offset, image_mask, image_files, image_dir)
                    for band in bands["bands"]
                ],
            )
        )

    return arrays


def get_image_metadata(
    image_dir: str, image_files: list[str]
) -> tuple[str, float, str, datetime] | None:
    image_metadata: dict[str, str] = {}
    metadata_txt: str | None = next(
        (file for file in image_files if file.endswith("MTL.txt")), None
    )
    if metadata_txt is None:
        return None

    with open(join(image_dir, metadata_txt), "r") as file:
        for line in file:
            if any([meta_name in line for meta_name in REQUIRED_METADATA]):
                meta_name = line.split("=")[0].strip()
                image_metadata[meta_name] = line.split("=")[1].strip().replace('"', "")
            if "LEVEL2_PROCESSING_RECORD" in line:
                break

    sun_elevation = float(image_metadata["SUN_ELEVATION"])
    date_acquired = image_metadata["DATE_ACQUIRED"]

    image_date = date_acquired.replace("_", "-")
    landsat_version = image_metadata["SPACECRAFT_ID"]
    scene_center_time = image_metadata["SCENE_CENTER_TIME"][:-2]
    utc_timestamp = f"{date_acquired} {scene_center_time}"
    time_start = datetime.strptime(utc_timestamp, "%Y-%m-%d %H:%M:%S.%f")

    return landsat_version, sun_elevation, image_date, time_start


def get_lat_lon(tif_meta: dict) -> tuple[NDArray, NDArray]:
    crs = tif_meta["crs"]
    transform = tif_meta["transform"]
    width = tif_meta["width"]
    height = tif_meta["height"]

    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = transform * (cols, rows)
    proj = pyproj.Transformer.from_crs(crs, "EPSG:4326")  # Convert to WGS84
    lats, lons = proj.transform(xs, ys)

    return lats, lons


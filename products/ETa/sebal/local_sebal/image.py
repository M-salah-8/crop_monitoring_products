#----------------------------------------------------------------------------------------#
#---------------------------------------//GEESEBAL//-------------------------------------#
#GEESEBAL - GOOGLE EARTH ENGINE APP FOR SURFACE ENERGY BALANCE ALGORITHM FOR LAND (SEBAL)
#CREATE BY: LEONARDO LAIPELT, RAFAEL KAYSER, ANDERSON RUHOFF AND AYAN FLEISCHMANN
#PROJECT - ET BRASIL https://etbrasil.org/
#LAB - HIDROLOGIA DE GRANDE ESCALA [HGE] website: https://www.ufrgs.br/hge/author/hge/
#UNIVERSITY - UNIVERSIDADE FEDERAL DO RIO GRANDE DO SUL - UFRGS
#RIO GRANDE DO SUL, BRAZIL

#DOI
#VERSION 0.1.1
#CONTACT US: leonardo.laipelt@ufrgs.br

#----------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------#

from os.path import dirname, exists, join
from os import makedirs, listdir
from glob import glob
from shutil import rmtree
from datetime import datetime

import rasterio
import numpy as np
# import pickle

from .masks import f_cloudMaskL_8_9_SR, f_albedoL_8_9
from .meteorology import get_meteorology
from .tools import (
    fexp_spec_ind,
    fexp_radlong_up,
    lst_dem_correction,
    fexp_radshort_down,
    fexp_radlong_down,
    fexp_radbalance,
    fexp_soil_heat,
    fexp_sensible_heat_flux,
    co_dem,
)
from .endmembers import fexp_cold_pixel, fexp_hot_pixel
from .evapotranspiration import fexp_et
from .download_meteorology import download_era5_hourly


# IMAGE FUNCTION

# ENDMEMBERS DEFAULT
# ALLEN ET AL. (2013)

def sebal_local(
    image_dir: str,
    local_data_dr: str,
    results_dr: str,
    p_top_NDVI: int = 5,
    p_coldest_Ts: int = 20,
    p_lowest_NDVI: int = 10,
    p_hottest_Ts: int = 20,
) -> None:

    # get image information
    local_data_dr = local_data_dr
    date_dir = dirname(image_dir)
    cal_bands_dr = join(image_dir, "calculated_bands")
    if exists(cal_bands_dr):
        rmtree(cal_bands_dr)
    makedirs(cal_bands_dr, exist_ok= True)
    results_dr = results_dr
    meta_names = ["LANDSAT_PRODUCT_ID", "SPACECRAFT_ID", "SUN_ELEVATION", "CLOUD_COVER", "SCENE_CENTER_TIME", "DATE_ACQUIRED"]
    meta: dict[str, str] = {}
    txt_meta: str | None = next(
        (file for file in glob(join(image_dir, "*")) if file.endswith("MTL.txt")), None
    )
    assert txt_meta is not None, "MTL.txt file dose not exists"
    with open(join(image_dir, txt_meta), "r") as file:
        for line in file:
            if any(meta_name in line for meta_name in meta_names):
                meta_name = line.split("=")[0].strip()
                meta[meta_name] = line.split("=")[1].strip().replace('"', '')
            if "LEVEL2_PROCESSING_RECORD" in line:
                break
    # _index=meta['LANDSAT_PRODUCT_ID']
    # cloud_cover=float(meta['CLOUD_COVER'])
    landsat_version = meta['SPACECRAFT_ID']
    sun_elevation = float(meta['SUN_ELEVATION'])
    utc_timestamp = f"{meta['DATE_ACQUIRED']} {meta['SCENE_CENTER_TIME'][:-2]}"
    time_start = datetime.strptime(utc_timestamp, "%Y-%m-%d %H:%M:%S.%f")
    date_string = meta['DATE_ACQUIRED']

    # LANDSAT IMAGE
    if landsat_version in ["LANDSAT_8", "LANDSAT_9"]:
        bands = {"SR_B1": "UB", "SR_B2": "B", "SR_B3": "GR", "SR_B4": "R",
                    "SR_B5": "NIR", "SR_B6": "SWIR_1", "SR_B7": "SWIR_2",
                    "ST_B10": "BRT", "QA_PIXEL": "pixel_qa"}
        image: dict[str, str] = {}
        for file in listdir(image_dir):
            if any(band in file for band in bands):
                band_name = "_".join(file.split("_")[-2:]).split(".")[0]
                image[bands[band_name]] = join(image_dir, file)
        ls_meta = rasterio.open(image["UB"]).meta
        ls_meta.update(dtype= np.float32, nodata= np.nan)
        res: tuple[float, float] = rasterio.open(image["UB"]).res
        # CLOUD REMOVAL
        f_cloudMaskL_8_9_SR(image, cal_bands_dr)

        # ALBEDO TASUMI ET AL. (2008) METHOD WITH KE ET AL. (2016) COEFFICIENTS
        f_albedoL_8_9(image, ls_meta, cal_bands_dr)

    else:
        print(f"This version is not supported: {landsat_version}")
        return

    # METEOROLOGY PARAMETERS
    download_era5_hourly(image["UB"], date_dir, time_start)
    t_air, ux, ur, rn24hobs = get_meteorology(image, time_start, date_dir, cal_bands_dr, ls_meta)

    # SRTM DATA ELEVATION
    srtm_elevation = join(local_data_dr, "strm_30m.tif")
    assert exists(srtm_elevation), "elevation file was not found"
    z_alt = co_dem(ls_meta.copy(), res, srtm_elevation, date_dir)

    # SPECTRAL IMAGES (NDVI, EVI, SAVI, LAI, T_LST, e_0, e_NB, long, lat)
    fexp_spec_ind(image, ls_meta, cal_bands_dr, results_dr, date_string)

    # LAND SURFACE TEMPERATURE
    lst_dem_correction(image, z_alt, t_air, ur, sun_elevation, time_start, time_start.hour,time_start.minute, ls_meta, cal_bands_dr)

    # COLD PIXEL
    _, n_Ts_cold = fexp_cold_pixel(image, p_top_NDVI, p_coldest_Ts)

    # INSTANTANEOUS OUTGOING LONG-WAVE RADIATION [W M-2]
    fexp_radlong_up(image, cal_bands_dr, ls_meta)

    # INSTANTANEOUS INCOMING SHORT-WAVE RADIATION [W M-2]
    fexp_radshort_down(image, z_alt, t_air,ur, sun_elevation, time_start, ls_meta, cal_bands_dr)

    # INSTANTANEOUS INCOMING LONGWAVE RADIATION [W M-2]
    fexp_radlong_down(image,  n_Ts_cold, cal_bands_dr, ls_meta)

    # INSTANTANEOUS NET RADIATON BALANCE [W M-2]
    fexp_radbalance(image, cal_bands_dr, ls_meta)

    # SOIL HEAT FLUX (G) [W M-2]
    fexp_soil_heat(image, cal_bands_dr, ls_meta)

    # HOT PIXEL
    d_hot_pixel=fexp_hot_pixel(image, p_lowest_NDVI, p_hottest_Ts)

    # SENSIBLE HEAT FLUX (H) [W M-2]
    fexp_sensible_heat_flux(image, ux, n_Ts_cold, d_hot_pixel, cal_bands_dr, ls_meta)

    # DAILY EVAPOTRANSPIRATION (ET_24H) [MM DAY-1]
    fexp_et(image, rn24hobs, cal_bands_dr, ls_meta, results_dr, date_string)

    rmtree(cal_bands_dr)

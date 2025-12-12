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
from shutil import rmtree

from util.logs import logger

from .funcs import get_image_metadata, get_image_tif_metadata, get_lat_lon, open_band, open_bands
from .masks import f_final_mask, f_albedoL_8_9
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
    prepare_sensible_heat_flux,
)
from .endmembers import fexp_cold_pixel, fexp_hot_pixel, hot_cold_pixels_helpers
from .evapotranspiration import fexp_et
from .download_meteorology import download_era5_hourly


REQUIRED_BANDS: dict[str, dict] = {
    "SR": {
        "bands": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
        "scale": 0.0000275,
        "offset": -0.2,
    },
    "ST": {
        "bands": ["ST_B10"],
        "scale": 0.00341802,
        "offset": 149,
    },
    "QA": {
        "bands": ["QA_PIXEL"],
        "scale": 1,
        "offset": 0,
    },
}


def sebal_local(
    image_dir: str,
    local_data_dr: str,
    results_dr: str,
    p_top_NDVI: int = 1,
    p_coldest_Ts: int = 1,
    p_lowest_NDVI: int = 1,
    p_hottest_Ts: int = 1,
) -> None:

    date_dir = dirname(image_dir)
    cal_bands_dr = join(image_dir, "calculated_bands")
    if exists(cal_bands_dr):
        rmtree(cal_bands_dr)
    makedirs(cal_bands_dr)
    image_files = listdir(image_dir)
    metadata_results = get_image_metadata(image_dir, image_files)
    if metadata_results is None:
        logger.error("Metadata was not found")
        return None
    landsat_version, sun_elevation, image_date, time_start = metadata_results

    if landsat_version not in ["LANDSAT_8", "LANDSAT_9"]:
        logger.error(f"This version is not supported: {landsat_version}")
        return None

    # Get the blue band to extract the image metadata
    in_blue_tif = REQUIRED_BANDS["SR"]["bands"][1]
    blue_band_check = [
        band_file for band_file in image_files if band_file.endswith(f"{in_blue_tif}.TIF")
    ]
    assert len(blue_band_check) == 1
    blue_tif = join(image_dir, blue_band_check[0])
    tif_meta, original_nodata, res, bounds = get_image_tif_metadata(blue_tif)

    b_blue_int = open_band(
        "SR_B2",
        image_files,
        image_dir,
    )
    b_pixel_qa = open_band(
        "QA_PIXEL",
        image_files,
        image_dir,
    )
    # CLOUD REMOVAL
    image_mask = f_final_mask(b_pixel_qa, b_blue_int, original_nodata)

    del(b_pixel_qa, b_blue_int)

    b_ca, b_blue, b_green, b_red, b_nir, b_swir_1, b_swir_2 = open_bands(
        REQUIRED_BANDS["SR"],
        image_mask,
        image_files,
        image_dir,
    )

    # ALBEDO TASUMI ET AL. (2008) METHOD WITH KE ET AL. (2016) COEFFICIENTS
    alfa = f_albedoL_8_9(b_ca, b_blue, b_green, b_red, b_nir, b_swir_1, b_swir_2)

    del(b_ca, b_swir_1, b_swir_2)

    lats, lons = get_lat_lon(tif_meta)

    # METEOROLOGY PARAMETERS
    download_era5_hourly(tif_meta["crs"], bounds, date_dir, time_start)
    t_air, ux, ur, rn24hobs = get_meteorology(
        alfa, lats, image_mask, time_start, date_dir, tif_meta, res
    )

    b_tirs_1 = open_bands(
        REQUIRED_BANDS["ST"],
        image_mask,
        image_files,
        image_dir,
    )[0]


    # SPECTRAL IMAGES (NDVI, EVI, SAVI, LAI, T_LST, e_0, e_NB)
    lst, ndwi, ndvi, lai, e_0, savi = fexp_spec_ind(
        b_blue, b_green, b_red, b_nir, b_tirs_1, tif_meta, results_dr, image_date
    )

    del(b_blue, b_green, b_red, b_nir, b_tirs_1)

    # SRTM DATA ELEVATION
    srtm_elevation = join(local_data_dr, "strm_30m.tif")
    assert exists(srtm_elevation), "elevation file was not found"
    z_alt, z_alt_file = co_dem(tif_meta.copy(), res, srtm_elevation, date_dir)

    # LAND SURFACE TEMPERATURE
    lst_dem = lst_dem_correction(
        lst,
        z_alt_file,
        z_alt,
        t_air,
        ur,
        lats,
        lons,
        sun_elevation,
        time_start,
        cal_bands_dr,
    )

    del (lats, lons)

    pos_ndvi, lst_nw = hot_cold_pixels_helpers(
        ndvi,
        ndwi,
        lst,
    )

    del (ndwi)

    # COLD PIXEL
    _, n_Ts_cold = fexp_cold_pixel(pos_ndvi, lst_nw, ndvi, p_top_NDVI, p_coldest_Ts)

    # INSTANTANEOUS OUTGOING LONG-WAVE RADIATION [W M-2]
    rl_up = fexp_radlong_up(lai, lst)

    del(lai)

    # INSTANTANEOUS INCOMING SHORT-WAVE RADIATION [W M-2]
    tao_sw, rs_down = fexp_radshort_down(z_alt, t_air, ur, sun_elevation, time_start)

    del (z_alt, t_air, ur, sun_elevation, time_start)

    # INSTANTANEOUS INCOMING LONGWAVE RADIATION [W M-2]
    rl_down = fexp_radlong_down(tao_sw,  n_Ts_cold, cal_bands_dr, tif_meta)

    del (tao_sw)

    # INSTANTANEOUS NET RADIATON BALANCE [W M-2]
    rn = fexp_radbalance(alfa, rs_down, rl_down, rl_up, e_0)

    del(e_0, rl_up, rs_down, rl_down)

    # SOIL HEAT FLUX (G) [W M-2]
    G = fexp_soil_heat(rn, ndvi, alfa, lst_dem)

    del (alfa)

    # HOT PIXEL
    d_hot_pixel = fexp_hot_pixel(
        pos_ndvi, lst, G, rn, ndvi, lst_nw, p_lowest_NDVI, p_hottest_Ts
    )

    del (pos_ndvi, lst, ndvi, lst_nw)

    # SENSIBLE HEAT FLUX (H) [W M-2]
    i_u200, i_ufric, i_zom, i_rah = prepare_sensible_heat_flux(
        savi,
        ux,
    )

    del (savi, ux)

    H = fexp_sensible_heat_flux(
        lst_dem,
        i_u200,
        i_ufric,
        i_zom,
        i_rah,
        n_Ts_cold,
        d_hot_pixel,
    )
    del (n_Ts_cold, d_hot_pixel)

    # DAILY EVAPOTRANSPIRATION (ET_24H) [MM DAY-1]
    fexp_et(rn, G, lst_dem, H, rn24hobs, tif_meta, results_dr, image_date)

    rmtree(cal_bands_dr)

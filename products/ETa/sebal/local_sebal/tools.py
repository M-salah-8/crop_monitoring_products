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

#PYTHON PACKAGES
from os.path import exists, join
from os import makedirs, remove
from math import pi
import subprocess
from datetime import datetime
from copy import deepcopy
from threading import Thread

from numpy.typing import NDArray
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np

from util.logs import logger

from .funcs import save_data


def _save_rbg(
    b_blue: NDArray[np.float32],
    b_green: NDArray[np.float32],
    b_red: NDArray[np.float32],
    meta: dict,
    results_dr: str,
    image_date: str,
) -> None:
    rgb = np.zeros([3, *b_blue.shape], np.uint8)
    for i, b in enumerate([b_red, b_green, b_blue]):
        rgb[i] = np.where(np.isnan(b), rgb[i], np.round(b * 255)).astype(np.uint8)
    rgb_meta = deepcopy(meta)
    rgb_meta.update(count=3, dtype="uint8", nodata=0)
    makedirs(join(results_dr, "rgb"), exist_ok=True)
    with rasterio.open(
        join(results_dr, "rgb", f"{image_date}.tif"), "w", **rgb_meta
    ) as dst:
        dst.write(rgb)


#SPECTRAL INDICES MODULE
def fexp_spec_ind(
    b_blue: NDArray[np.float32],
    b_green: NDArray[np.float32],
    b_red: NDArray[np.float32],
    b_nir: NDArray[np.float32],
    b_tirs_1: NDArray[np.float32],
    meta: dict,
    results_dr: str,
    image_date: str,
) -> tuple[NDArray[np.float32], ...]:
    logger.info("calculate spectral indices")

    Thread(
        target=_save_rbg,
        args=(
            b_blue,
            b_green,
            b_red,
            meta,
            results_dr,
            image_date,
        ),
    ).start()

    # NORMALIZED DIFFERENCE VEGETATION INDEX (NDVI)
    ndvi = (b_nir - b_red) / (b_nir + b_red)
    ndvi_dst = join(results_dr, "ndvi")
    makedirs(ndvi_dst, exist_ok=True)

    Thread(
        target=save_data,
        args=(ndvi_dst, meta, ndvi, image_date),
    ).start()

    # # ENHANCED VEGETATION INDEX (EVI)
    # evi = 2.5 * ((b_nir - b_red) / (b_nir + (6 * b_red) - (7.5 * b_blue) + 1))

    # SOIL ADHUSTED VEGETATION INDEX (SAVI)
    savi = ((1 + 0.5) * (b_nir - b_red)) / (0.5 + (b_nir + b_red))

    # NORMALIZED DIFFERENCE WATER INDEX (NDWI)
    ndwi = (b_green - b_nir) / (b_green + b_nir)

    savi1 = savi.copy()
    savi1[savi1 > 0.689] = 0.689

    # LEAF AREA INDEX (LAI)
    lai = -(np.log((0.69 - savi1) / 0.59) / 0.91)

    NDVI_adjust = ndvi.copy()
    NDVI_adjust[NDVI_adjust < 0] = 0
    NDVI_adjust[NDVI_adjust > 1] = 1
    fipar = NDVI_adjust * 1 - 0.05
    fipar[fipar < 0] = 0
    fipar[fipar > 1] = 1
    del NDVI_adjust

    # BROAD-BAND SURFACE EMISSIVITY (e_0)
    e_0 = 0.95 + 0.01 * lai
    e_0[lai > 3] = 0.98

    # NARROW BAND TRANSMISSIVITY (e_NB)
    e_NB = 0.97 + (0.0033 * lai)
    e_NB[lai > 3] = 0.98
    log_eNB = np.log(e_NB)

    # LAND SURFACE TEMPERATURE (LST) [K]
    comp_onda = 1.115e-05
    lst = b_tirs_1 / (1 + ((comp_onda * b_tirs_1 / 1.438e-02) * log_eNB))

    return lst, ndwi, ndvi, lai, e_0, savi


#LAND SURFACE TEMPERATURE WITH DEM CORRECTION AND ASPECT/SLOPE
#JAAFAR AND AHMAD (2020)
#PYSEBAL (BASTIAANSSEN) Reference?
def lst_dem_correction(
    lst: NDArray[np.float32],
    z_alt_file: str,
    z_alt: NDArray[np.float32],
    t_air: NDArray[np.float32],
    ur: NDArray[np.float32],
    lats: NDArray,
    lons: NDArray,
    sun_elevation: float,
    time_start: datetime,
    cal_bands_dr: str,
) -> NDArray[np.float32]:

    #SOLAR CONSTANT [W M-2]
    gsc = 1367

    #DAY OF YEAR
    doy = time_start.timetuple().tm_yday

    #INVERSE RELATIVE  DISTANCE EARTH-SUN
    d1 = 2 *pi / 365
    d2 = d1 * doy
    d3 = np.cos(d2)
    dr = 1 + (0.033 * d3)

    #SOLAR ZENITH ANGLE OVER A HORZONTAL SURFACE
    solar_zenith = 90 - sun_elevation
    degree2radian = 0.01745
    solar_zenith_radians = solar_zenith * degree2radian
    cos_theta = np.cos(solar_zenith_radians)

    # COS ZENITH ANGLE SUN ELEVATION #ALLEN ET AL. (2006)
    subprocess.run(
        [
            "gdaldem",
            "slope",
            z_alt_file,
            join(cal_bands_dr, "slope.tif"),
            "-of", "GTiff",
            "-b", "1",
            "-s", "1.0",
            "-compute_edges",
        ]
    )
    with rasterio.open(join(cal_bands_dr, "slope.tif")) as dataset:
        slope = dataset.read(1)

    subprocess.run(
        [
            "gdaldem",
            "aspect",
            z_alt_file,
            join(cal_bands_dr, "aspect.tif"),
            "-of", "GTiff",
            "-b", "1",
            "-compute_edges",
        ]
    )
    with rasterio.open(join(cal_bands_dr, "aspect.tif")) as dataset:
        aspect = dataset.read(1)

    B = (360 / 365) * (doy - 81)
    delta = np.arcsin(np.sin(23.45 * degree2radian)) * np.sin(B * degree2radian)
    s = slope * degree2radian
    gamma = (aspect - 180) * degree2radian
    phi = lats * degree2radian
    del(slope, aspect)

    # CONSTANTS ALLEN ET AL. (2006)
    a = (np.sin(delta) * np.cos(phi) * np.sin(s) * (np.cos(gamma))) - (
        np.sin(delta) * (np.sin(phi) * np.cos(s))
    )
    b = (np.cos(delta) * np.cos(phi) * np.cos(s)) + (
        np.cos(delta) * (np.sin(phi) * np.sin(s) * np.cos(gamma))
    )
    c = np.cos(delta) * np.sin(s) * np.sin(gamma)
    del (delta, s, gamma, phi)

    #GET IMAGE CENTROID
    center_x = int((lons.shape[0] - 1) / 2)
    center_y = int((lons.shape[1] - 1) / 2)
    longitude_center = lons[center_x, center_y]

    #DELTA GTM
    delta_gtm = int(longitude_center / 15)

    min_to_hour = time_start.minute / 60

    #LOCAL HOUR TIME
    local_hour_time = time_start.hour + delta_gtm + min_to_hour

    hour_a = (local_hour_time - 12) * 15

    w = hour_a * degree2radian

    cos_zn = -a +b*np.cos(w) +c*np.sin(w)
    del(a, b, c)

    #ATMOSPHERIC PRESSURE [KPA]
    #SHUTTLEWORTH (2012)
    pres = 101.3 * ((293 - (0.0065 * z_alt))/ 293) ** 5.26

    #SATURATION VAPOR PRESSURE (es) [KPA]
    es = 0.6108 *(np.exp((17.27 * t_air) / (t_air + 237.3)))

    #ACTUAL VAPOR PRESSURE (ea) [KPA]
    ea = es * ur / 100

    #WATER IN THE ATMOSPHERE [mm]
    #Garrison and Adler (1990)
    W = (0.14 * ea * pres) + 2.1

    del(es, ea)

    # BROAD-BAND ATMOSPHERIC TRANSMISSIVITY (tao_sw)
    # ASCE-EWRI (2005)
    # TODO: duplicated code
    tao_sw = 0.35 + 0.627 * np.exp(
        ((-0.00146 * pres) / (1 * cos_theta)) - (0.075 * (W / cos_theta) ** 0.4)
    )

    # AIR DENSITY [KG M-3]
    air_dens = (1000 * pres) / (1.01 * lst * 287)

    del (pres, W)

    # TEMPERATURE LAPSE RATE (0.0065)
    Temp_lapse_rate= 0.0065

    # LAND SURFACE TEMPERATURE CORRECTION DEM [K]
    Temp_corr = lst + (z_alt * Temp_lapse_rate)

    # LAND SURFACE TEMPERATURE WITH ASPECT/SLOPE CORRECTION [K]
    lst_dem = Temp_corr + (
        gsc * dr * tao_sw * cos_zn - gsc * dr * tao_sw * cos_theta
    ) / (air_dens * 1004 * 0.050)

    return lst_dem.astype(np.float32)


# INSTANTANEOUS OUTGOING LONG-WAVE RADIATION (Rl_up) [W M-2]
def fexp_radlong_up(
    lai: NDArray[np.float32], lst: NDArray[np.float32]
) -> NDArray[np.float32]:
    # BROAD-BAND SURFACE THERMAL EMISSIVITY
    # TASUMI ET AL. (2003)
    #ALLEN ET AL. (2007)
    emi = 0.95 + (0.01 * lai)
    #LAI
    emi[lai > 3] = 0.98
    stefBol = np.full(emi.shape, 5.67e-8)

    rl_up = emi * stefBol * (lst ** 4)
    return rl_up


#INSTANTANEOUS INCOMING SHORT-WAVE RADIATION (Rs_down) [W M-2]
def fexp_radshort_down(
    z_alt: NDArray,
    t_air: NDArray[np.float32],
    ur: NDArray[np.float32],
    sun_elevation: float,
    time_start: datetime,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:

    #SOLAR CONSTANT
    gsc = 1367 #[W M-2]

    #DAY OF THE YEAR
    doy = time_start.timetuple().tm_yday

    #INVERSE RELATIVE  DISTANCE EARTH-SUN
    d1 =  (2 * pi) / 365
    d2 = d1 * doy
    d3 = np.cos(d2)
    dr = 1 + (0.033 * d3)

    #ATMOSPHERIC PRESSURE [KPA]
    #SHUTTLEWORTH (2012)
    pres = 101.3 * ((293 - (0.0065 * z_alt))/ 293) ** 5.26

    #SATURATION VAPOR PRESSURE (es) [KPA]
    es = 0.6108 * (np.exp( (17.27 * t_air) / (t_air + 237.3)))

    #ACTUAL VAPOR PRESSURE (ea) [KPA]
    ea = es * ur / 100

    #WATER IN THE ATMOSPHERE [mm]
    #GARRISON AND ADLER (1990)
    W = (0.14 * ea * pres) + 2.1

    #SOLAR ZENITH ANGLE OVER A HORIZONTAL SURFACE
    solar_zenith = 90 - sun_elevation
    degree2radian = 0.01745
    solar_zenith_radians = solar_zenith * degree2radian
    cos_theta = np.cos(solar_zenith_radians)

    #BROAD-BAND ATMOSPHERIC TRANSMISSIVITY (tao_sw)
    # ASCE-EWRI (2005)
    tao_sw = 0.35 + 0.627 * np.exp(
        ((-0.00146 * pres) / (1 * cos_theta)) - (0.075 * (W / cos_theta) ** 0.4)
    )

    #INSTANTANEOUS SHORT-WAVE RADIATION (Rs_down) [W M-2]
    rs_down = gsc * cos_theta * tao_sw * dr

    return tao_sw, rs_down


    #INSTANTANEOUS INCOMING LONGWAVE RADIATION (Rl_down) [W M-2]
    #ALLEN ET AL (2007)
def fexp_radlong_down(
    tao_sw: NDArray[np.float32], n_Ts_cold: float, cal_bands_dr: str, meta: dict
) -> NDArray[np.float32]:

    log_taosw = np.log(tao_sw)
    rl_down = (0.85 * (- log_taosw) ** 0.09) * 5.67e-8 * (n_Ts_cold ** 4)

    return rl_down

    # INSTANTANEOUS NET RADIATON BALANCE (Rn) [W M-2]


def fexp_radbalance(
    alfa: NDArray[np.float32],
    rs_down: NDArray[np.float32],
    rl_down: NDArray[np.float32],
    rl_up: NDArray[np.float32],
    e_0: NDArray[np.float32],
) -> NDArray[np.float32]:
    rn = ((1 - alfa) * rs_down) + rl_down - rl_up - ((1 - e_0) * rl_down)
    return rn

    # SOIL HEAT FLUX (G) [W M-2]
    # BASTIAANSSEN (2000)


def fexp_soil_heat(
    rn: NDArray[np.float32],
    ndvi: NDArray[np.float32],
    alfa: NDArray[np.float32],
    lst_dem: NDArray[np.float32],
) -> NDArray[np.float32]:
    G = rn * (lst_dem - 273.15) * (0.0038 + (0.0074 * alfa)) * (1 - 0.98 * (ndvi**4))
    return G


def prepare_sensible_heat_flux(
    savi: NDArray[np.float32],
    ux: NDArray[np.float32],
) -> tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]
]:
    # VEGETATION HEIGHTS  [M]
    n_veg_hight = 3

    # WIND SPEED AT HEIGHT Zx [M]
    n_zx = 2

    # BLENDING HEIGHT [M]
    n_hight = 200

    # VON KARMAN'S CONSTANT
    n_K = 0.41

    # MOMENTUM ROUGHNESS LENGHT (ZOM) AT THE WEATHER STATION [M]
    # BRUTSAERT (1982)
    n_zom = n_veg_hight * 0.12

    # Z1 AND Z2 ARE HEIGHTS [M] ABOVE THE ZERO PLANE DISPLACEMENT
    # OF THE VEGETATION
    z1 = 0.1
    z2 = 2

    # FRICTION VELOCITY AT WEATHER STATION [M S-1]
    i_ufric_ws = (n_K * ux) / np.log(n_zx / n_zom)

    # WIND SPEED AT BLENDING HEIGHT AT THE WEATHER STATION [M S-1]
    i_u200 = i_ufric_ws * (np.log(n_hight / n_zom) / n_K)
    del i_ufric_ws

    # MOMENTUM ROUGHNESS LENGHT (ZOM) FOR EACH PIXEL [M]
    i_zom = np.exp((5.62 * (savi)) - 5.809)

    # FRICTION VELOCITY FOR EACH PIXEL  [M S-1]
    i_ufric = (n_K * i_u200) / (np.log(n_hight / n_zom))

    # AERODYNAMIC RESISTANCE TO HEAT TRANSPORT (rah) [S M-1]
    i_rah = (np.log(z2 / z1)) / (i_ufric * 0.41)

    return i_u200, i_ufric, i_zom, i_rah


# SENSIBLE HEAT FLUX (H) [W M-2]
def fexp_sensible_heat_flux(
    lst_dem: NDArray[np.float32],
    i_u200: NDArray[np.float32],
    i_ufric: NDArray[np.float32],
    i_zom: NDArray[np.float32],
    i_rah: NDArray[np.float32],
    n_Ts_cold: float,
    d_hot_pixel: dict,
) -> NDArray[np.float32]:

    # BLENDING HEIGHT [M]
    n_hight = 200

    # AIR SPECIFIC HEAT [J kg-1/K-1]
    n_Cp = 1004

    #Z1 AND Z2 ARE HEIGHTS [M] ABOVE THE ZERO PLANE DISPLACEMENT
    #OF THE VEGETATION
    z1= 0.1
    z2= 2

    # TS HOT PIXEL
    n_Ts_hot = d_hot_pixel["temp"]
    # G HOT PIXEL
    n_G_hot = d_hot_pixel["G"]
    # RN HOT PIXEL
    n_Rn_hot = d_hot_pixel["Rn"]

    # AIR DENSITY HOT PIXEL
    n_ro_hot = (-0.0046 * n_Ts_hot) + 2.5538

    # NEAR SURFACE TEMPERATURE DIFFERENCE IN COLD PIXEL (dT= tZ1-tZ2)
    n_dT_cold = 0

    i_lst_med = lst_dem

    #========ITERATIVE PROCESS=========#

    #SENSIBLE HEAT FLUX AT THE HOT PIXEL (H_hot)
    n_H_hot = n_Rn_hot - n_G_hot

    #ITERATIVE VARIABLES
    n = 1
    n_dif = 1
    list_dif = []
    list_dT_hot = []
    list_rah_hot = []
    list_coef_a = []
    list_coef_b = []

    # NUMBER OF ITERATIVE STEPS: 15
    # CAN BE CHANGED, BUT BE AWARE THAT
    # A MINIMUM NUMBER OF ITERATIVE PROCESSES
    # IS NECESSARY TO ACHIEVE RAH AND H ESTIMATIONS

    #========INIT ITERATION========#
    for n in range(15):
        # AERODYNAMIC RESISTANCE TO HEAT TRANSPORT
        # IN HOT PIXEL
        n_rah_hot = i_rah[d_hot_pixel['index'][0], d_hot_pixel['index'][1]]

        # NEAR SURFACE TEMPERATURE DIFFERENCE IN HOT PIXEL (dT= Tz1-Tz2)  [K]
        # dThot= Hhot*rah/(ρCp)
        n_dT_hot = (n_H_hot * n_rah_hot) / (n_ro_hot * n_Cp)

        # dT =  aTs + b
        # ANGULAR COEFFICIENT
        n_coef_a = (n_dT_cold - n_dT_hot) / (n_Ts_cold - n_Ts_hot)

        # LINEAR COEFFICIENT
        n_coef_b = n_dT_hot - (n_coef_a * n_Ts_hot)

        # dT FOR EACH PIXEL [K]
        i_dT_int = (n_coef_a * i_lst_med) + n_coef_b

        # AIR TEMPERATURE (TA) FOR EACH PIXEL (TA=TS-dT) [K]
        i_Ta = i_lst_med - i_dT_int

        # AIR DENSITY (ro) [KM M-3]
        i_ro = (-0.0046 * i_Ta) + 2.5538

        del (i_Ta)

        # SENSIBLE HEAT FLUX (H) FOR EACH PIXEL  [W M-2]
        i_H_int = (i_ro * n_Cp * i_dT_int) / i_rah

        # # GET VALUE
        # n_H_int = i_H_int[d_hot_pixel["index"][0], d_hot_pixel["index"][1]]

        # MONIN-OBUKHOV LENGTH (L)
        # FOR STABILITY CONDITIONS OF THE ATMOSPHERE IN THE ITERATIVE PROCESS
        i_L_int = -(i_ro * n_Cp * (i_ufric**3) * i_lst_med) / (0.41 * 9.81 * i_H_int)
        del i_H_int

        # STABILITY CORRECTIONS FOR MOMENTUM AND HEAT TRANSPORT
        # PAULSON (1970)
        # WEBB (1970)

        # Start with height = 200 m then repeat for 2 m and 0.1 m
        # STABILITY CORRECTIONS FOR STABLE CONDITIONS
        i_psim_200 = -5 * (200 / i_L_int)
        # FOR DIFFERENT HEIGHT
        i_x = (1 - (16 * (200 / i_L_int))) ** 0.25
        # STABILITY CORRECTIONS FOR UNSTABLE CONDITIONS
        i_psi = (
            2 * np.log((1 + i_x) / 2)
            + np.log((1 + i_x**2) / 2)
            - 2 * np.arctan(i_x)
            + 0.5 * pi
        )
        # FOR EACH PIXEL
        i_psim_200 = np.where(i_L_int < 0, i_psi, i_psim_200)
        i_psim_200 = np.where(i_L_int == 0, np.float32(0), i_psim_200)
        pass

        # for 2 m
        i_psih_2 = -5 * (2 / i_L_int)
        i_x = (1 - (16 * (2 / i_L_int))) ** 0.25
        i_psi = 2 * np.log((1 + i_x**2) / 2)
        i_psih_2 = np.where(i_L_int < 0, i_psi, i_psih_2)
        i_psih_2 = np.where(i_L_int == 0, np.float32(0), i_psih_2)

        # for 0.1 meter
        i_psih_01 = -5 * (0.1 / i_L_int)
        i_x = (1 - (16 * (0.1 / i_L_int))) ** 0.25
        i_psi = 2 * np.log((1 + i_x**2) / 2)
        i_psih_01 = np.where(i_L_int < 0, i_psi, i_psih_01)
        i_psih_01 = np.where(i_L_int == 0, np.float32(0), i_psih_01)

        del (i_L_int, i_psi, i_x)

        # if n==1:
        #     i_psim_200_exp = i_psim_200
        #     i_psih_2_exp = i_psih_2
        #     i_psih_01_exp = i_psih_01
        #     i_L_int_exp = i_L_int
        #     i_H_int_exp = i_H_int
        #     i_dT_int_exp = i_dT_int
        #     i_rah_exp = i_rah

        # CORRECTED VALUE FOR THE FRICTION VELOCITY (i_ufric) [M S-1]
        i_ufric = (i_u200 * 0.41) / (np.log(n_hight / i_zom) - i_psim_200)

        # CORRECTED VALUE FOR THE AERODYNAMIC RESISTANCE TO THE HEAT TRANSPORT (rah) [S M-1]
        i_rah = (np.log(z2 / z1) - i_psih_2 + i_psih_01) / (i_ufric * 0.41)

        del (i_psih_01, i_psih_2, i_psim_200)

        if n == 0:
            n_dT_hot_old = n_dT_hot
            n_rah_hot_old = n_rah_hot
            n_dif = 1

        if n > 0:
            n_dT_hot_abs = abs(n_dT_hot)
            n_dT_hot_old_abs = abs(n_dT_hot_old)
            n_rah_hot_abs = abs(n_rah_hot)
            n_rah_hot_old_abs = abs(n_rah_hot_old)
            n_dif= abs(n_dT_hot_abs - n_dT_hot_old_abs + n_rah_hot_abs - n_rah_hot_old_abs)
            n_dT_hot_old = n_dT_hot
            n_rah_hot_old = n_rah_hot

        logger.info(f"n = {n} {n_dif} {n_coef_a} {n_coef_b} {n_dT_hot} {n_rah_hot}")

        # INSERT EACH ITERATION VALUE INTO A LIST
        list_dif.append(n_dif)
        list_coef_a.append(n_coef_a)
        list_coef_b.append(n_coef_b)
        list_dT_hot.append(n_dT_hot)
        list_rah_hot.append(n_rah_hot)

    # =========END ITERATION =========#

    # GET FINAL rah, dT AND H
    i_rah_final = i_rah  # [SM-1]

    i_dT_final = i_dT_int  # [K]

    i_H_final = (i_ro * n_Cp * i_dT_final) / i_rah_final  # [W M-2]

    return i_H_final


# coregister strm data
def co_dem(
    meta: dict, res: tuple[float, float], srtm_elevation: str, ls_data_dr: str
) -> tuple[NDArray, str]:
    with rasterio.open(srtm_elevation) as src:
        dem, _ = reproject(
            src.read(1),
            destination=np.zeros([meta["height"], meta["width"]]),
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=-9999,
            dst_transform=meta["transform"],
            dst_crs=meta["crs"],
            dst_nodata=-9999,
            dst_resolution=res,
            resampling=Resampling.bilinear,
        )
    strm_file = join(ls_data_dr, "strm_30m_co.tif")
    if exists(strm_file):
        remove(strm_file)
    dem_meta = deepcopy(meta)
    dem_meta.update(
        nodata= -9999,
        dtype= "int16"
    )
    strm_file = save_data(ls_data_dr, dem_meta, dem, "strm_30m_co")
    return dem, strm_file

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

from numpy.typing import NDArray
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np

from .funcs import save_to_image, save_data


#SPECTRAL INDICES MODULE
def fexp_spec_ind(
    image: dict[str, str],
    meta: dict,
    cal_bands_dr: str,
    results_dr: str,
    date_string: str,
) -> None:
    image_mask: NDArray[np.bool_] = np.load(image["MASK"])
    sr_bands = ["B", "R", "NIR", "GR"]
    st_bands = ["BRT"]
    arrays: dict[str, NDArray[np.float32]] = {}
    array = None
    for band in sr_bands:
        with rasterio.open(image[band]) as src:
            array = src.read(1).astype(np.float32)
            array[image_mask] = np.nan
            arrays[band] = array.copy() * 0.0000275 - 0.2
    for band in st_bands:
        with rasterio.open(image[band]) as src:
            array = src.read(1).astype(np.float32)
            array[image_mask] = np.nan
            arrays[band] = array.copy() * 0.00341802 + 149
    del array
    rgb = np.zeros([3,*arrays["B"].shape])
    rgb[0] = arrays["R"].copy() * 255
    rgb[1] = arrays["GR"].copy() * 255
    rgb[2] = arrays["B"].copy() * 255
    rgb_meta = meta.copy()
    rgb_meta.update(count=3, dtype="uint8", nodata=0)
    makedirs(join(results_dr, "rgb"), exist_ok=True)
    with rasterio.open(join(results_dr, "rgb", f"rgb_{date_string}.tif"), "w", **rgb_meta) as dst:
        dst.write(rgb)
    del(rgb, rgb_meta)

    #NORMALIZED DIFFERENCE VEGETATION INDEX (NDVI)
    ndvi = (arrays["NIR"] - arrays["R"]) / (arrays["NIR"] + arrays["R"])
    save_to_image(image, cal_bands_dr, meta, ndvi, "ndvi")
    output_dr = join(results_dr, "ndvi")
    makedirs(output_dr, exist_ok=True)
    save_to_image(image, output_dr, meta, ndvi, f"ndvi_{date_string}")

    #ENHANCED VEGETATION INDEX (EVI)
    evi = 2.5 * (
        (arrays["NIR"] - arrays["R"])
        / (arrays["NIR"] + (6 * arrays["R"]) - (7.5 * arrays["B"]) + 1)
    )
    save_to_image(image, cal_bands_dr, meta, evi, "evi")
    del(evi)

    #SOIL ADHUSTED VEGETATION INDEX (SAVI)
    savi = ((1 + 0.5)*(arrays["NIR"] - arrays["R"])) / (0.5 + (arrays["NIR"] + arrays["R"]))
    save_to_image(image, cal_bands_dr, meta, savi, "savi")

    #NORMALIZED DIFFERENCE WATER INDEX (NDWI)
    ndwi = (arrays["GR"] - arrays["NIR"]) / (arrays["GR"] + arrays["NIR"])
    save_to_image(image, cal_bands_dr, meta, ndwi, 'ndwi')
    del(arrays["NIR"], arrays["GR"], arrays["R"], arrays["B"])

    savi1 = savi.copy()
    savi1[savi1 > 0.689] = 0.689
    del(savi)

    #LEAF AREA INDEX (LAI)
    lai = -(np.log((0.69-savi1)/0.59)/0.91)
    save_to_image(image, cal_bands_dr, meta, lai, "lai")
    del(savi1)

    NDVI_adjust = ndvi.copy()
    NDVI_adjust[NDVI_adjust < 0] = 0
    NDVI_adjust[NDVI_adjust > 1] = 1
    fipar = NDVI_adjust * 1 - 0.05
    fipar[fipar < 0] = 0
    fipar[fipar > 1] = 1
    save_to_image(image, cal_bands_dr, meta, fipar, "fipar")
    del(fipar, NDVI_adjust)

    #BROAD-BAND SURFACE EMISSIVITY (e_0)
    e_0 = 0.95 + 0.01 * lai
    e_0[lai > 3] = 0.98
    save_to_image(image, cal_bands_dr, meta, e_0, "e_0")
    del(e_0)

    #NARROW BAND TRANSMISSIVITY (e_NB)
    e_NB = 0.97 + (0.0033 * lai)
    e_NB[lai > 3] = 0.98
    save_to_image(image, cal_bands_dr, meta, e_NB, "e_NB")
    log_eNB = np.log(e_NB)

    #LAND SURFACE TEMPERATURE (LST) [K]
    comp_onda = 1.115e-05
    lst = arrays["BRT"] / (1+((comp_onda * arrays["BRT"] / 1.438e-02) * log_eNB))
    save_to_image(image, cal_bands_dr, meta, lst, "T_LST")
    del(log_eNB)

    #RESCALED BRIGHTNESS TEMPERATURE
    brt_r = arrays["BRT"]
    save_to_image(image, cal_bands_dr, meta, brt_r, "BRT_R")
    del(brt_r)

    #FOR FUTHER USE
    pos_ndvi = ndvi.copy()
    pos_ndvi[pos_ndvi <= 0] = np.nan
    save_to_image(image, cal_bands_dr, meta, pos_ndvi, "POS_NDVI")
    ndvi_neg =  pos_ndvi * -1
    save_to_image(image, cal_bands_dr, meta, ndvi_neg, "NDVI_NEG")
    int_ = np.full(ndvi.shape, 1, np.float32)
    int_[np.isnan(ndvi)] = np.nan
    save_to_image(image, cal_bands_dr, meta, int_, "INT")
    sd_ndvi = np.full(ndvi.shape, 1, np.float32)
    sd_ndvi[np.isnan(ndvi)] = np.nan
    save_to_image(image, cal_bands_dr, meta, sd_ndvi, "SD_NDVI")
    del(ndvi, pos_ndvi, ndvi_neg, int_, sd_ndvi)
    lst_neg = lst.copy() * -1
    save_to_image(image, cal_bands_dr, meta, lst_neg, "LST_NEG")
    del(lst_neg)
    lst_nw = lst.copy()
    lst_nw[ndwi > 0] = np.nan
    save_to_image(image, cal_bands_dr, meta, lst_nw, "LST_NW")

    #ADD BANDS
    del(arrays, lst_nw, lst)


#LAND SURFACE TEMPERATURE WITH DEM CORRECTION AND ASPECT/SLOPE
#JAAFAR AND AHMAD (2020)
#PYSEBAL (BASTIAANSSEN) Reference?
def lst_dem_correction(
    image: dict[str, str],
    z_alt: str,
    t_air: str,
    ur: str,
    sun_elevation: float,
    time_start: datetime,
    hour: int,
    minuts: int,
    meta: dict,
    cal_bands_dr: str,
) -> None:
    image_mask: NDArray[np.bool_] = np.load(image["MASK"])
    arrays: dict[str, NDArray[np.float32]] = {}
    array = None
    with rasterio.open(z_alt) as src:
        array = src.read(1).astype(np.float32)
        array[image_mask] = np.nan
        arrays["Z_ALT"] = array.copy()
    with rasterio.open(t_air) as src:
        array = src.read(1).astype(np.float32)
        array[image_mask] = np.nan
        arrays["T_AIR"] = array.copy()
    with rasterio.open(ur) as src:
        array = src.read(1).astype(np.float32)
        array[image_mask] = np.nan
        arrays["UR"] = array.copy()

    bands = ["T_LST", "NDWI"]
    for band in bands:
        with rasterio.open(image[band]) as src:
            array = src.read(1).astype(np.float32)
            array[image_mask] = np.nan
            arrays[band] = array.copy()
    bands = ["LATITUDE", "LONGITUDE"]   # do not mask
    for band in bands:
        with rasterio.open(image[band]) as src:
            array = src.read(1).astype(np.float32)
            arrays[band] = array.copy()
    del array

    #SOLAR CONSTANT [W M-2]
    gsc = 1367

    #DAY OF YEAR
    doy = time_start.timetuple().tm_yday

    #INVERSE RELATIVE  DISTANCE EARTH-SUN
    d1 = 2 *pi / 365
    d2 = d1 * doy
    d3 = np.cos(d2)
    dr = 1 + (0.033 * d3)

    #ATMOSPHERIC PRESSURE [KPA]
    #SHUTTLEWORTH (2012)
    pres = 101.3 * ((293 - (0.0065 * arrays["Z_ALT"]))/ 293) ** 5.26

    #SATURATION VAPOR PRESSURE (es) [KPA]
    es = 0.6108 *(np.exp((17.27 * arrays["T_AIR"]) / (arrays["T_AIR"] + 237.3)))

    #ACTUAL VAPOR PRESSURE (ea) [KPA]
    ea = es * arrays["UR"] / 100

    #WATER IN THE ATMOSPHERE [mm]
    #Garrison and Adler (1990)
    W = (0.14 * ea * pres) + 2.1
    del(es, ea)

    #SOLAR ZENITH ANGLE OVER A HORZONTAL SURFACE
    solar_zenith = 90 - sun_elevation
    degree2radian = 0.01745
    solar_zenith_radians = solar_zenith * degree2radian
    cos_theta = np.cos(solar_zenith_radians)

    #BROAD-BAND ATMOSPHERIC TRANSMISSIVITY (tao_sw)
    #ASCE-EWRI (2005)
    tao_sw = 0.35 + 0.627 * np.exp(((-0.00146 * pres)/(1 * cos_theta)) - (0.075 * (W / cos_theta)**0.4))

    #AIR DENSITY [KG M-3]
    air_dens = (1000* pres)/(1.01*arrays["T_LST"]*287)

    #TEMPERATURE LAPSE RATE (0.0065)
    Temp_lapse_rate= 0.0065

    #LAND SURFACE TEMPERATURE CORRECTION DEM [K]
    Temp_corr= arrays["T_LST"] + (arrays["Z_ALT"] * Temp_lapse_rate)

    # COS ZENITH ANGLE SUN ELEVATION #ALLEN ET AL. (2006)
    subprocess.run(
        [
            "gdaldem",
            "slope",
            z_alt,
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
            z_alt,
            join(cal_bands_dr, "aspect.tif"),
            "-of", "GTiff",
            "-b", "1",
            "-compute_edges",
        ]
    )
    with rasterio.open(join(cal_bands_dr, "aspect.tif")) as dataset:
        aspect = dataset.read(1)

    B = (360 / 365) * (doy - 81)
    delta = (np.arcsin(np.sin(23.45 * degree2radian)) * np.sin(B * degree2radian))
    s = slope * degree2radian
    gamma = (aspect - 180) * degree2radian
    phi = arrays["LATITUDE"] * degree2radian
    del(slope, aspect)

    #CONSTANTS ALLEN ET AL. (2006)
    a = (np.sin(delta) * np.cos(phi) * np.sin(s) * (np.cos(gamma))) - (np.sin(delta) * (np.sin(phi) * np.cos(s)))
    b = (np.cos(delta) * np.cos(phi) * np.cos(s)) + (np.cos(delta) * (np.sin(phi) * np.sin(s) * np.cos(gamma)))
    c = (np.cos(delta) * np.sin(s) * np.sin(gamma))
    del(delta, s, gamma, phi)

    #GET IMAGE CENTROID
    center_x = int((arrays["LONGITUDE"].shape[0] - 1) / 2)
    center_y = int((arrays["LONGITUDE"].shape[1] - 1) / 2)
    longitude_center= arrays["LONGITUDE"][center_x, center_y]

    #DELTA GTM
    delta_gtm = int(longitude_center / 15)

    min_to_hour= minuts / 60

    #LOCAL HOUR TIME
    local_hour_time = hour + delta_gtm + min_to_hour

    hour_a = (local_hour_time - 12) * 15

    w = hour_a * degree2radian

    cos_zn = -a +b*np.cos(w) +c*np.sin(w)
    del(a, b, c)

    #LAND SURFACE TEMPERATURE WITH ASPECT/SLOPE CORRECTION [K]
    TS_DEM= (Temp_corr + (gsc * dr * tao_sw * cos_zn -gsc * dr * tao_sw * cos_theta) / (air_dens * 1004 * 0.050))
    save_to_image(image, cal_bands_dr, meta, TS_DEM, "T_LST_DEM")

    #MASKS FOR SELECT PRE-CANDIDATES PIXELS
    lst_neg = TS_DEM * -1
    save_to_image(image, cal_bands_dr, meta, lst_neg, "LST_neg")
    ndwi = arrays["NDWI"]
    lst_nw = TS_DEM.copy()
    lst_nw[ndwi > 0] = np.nan
    save_to_image(image, cal_bands_dr, meta, lst_nw, "LST_NW")

    del(arrays, air_dens, Temp_corr, W, tao_sw, lst_nw, ndwi, lst_neg, TS_DEM, cos_zn)


#INSTANTANEOUS OUTGOING LONG-WAVE RADIATION (Rl_up) [W M-2]
def fexp_radlong_up(image: dict[str, str], cal_bands_dr: str, meta: dict) -> None:
    image_mask: NDArray[np.bool_] = np.load(image["MASK"])
    #BROAD-BAND SURFACE THERMAL EMISSIVITY
    #TASUMI ET AL. (2003)
    #ALLEN ET AL. (2007)
    bands = ['LAI', 'T_LST']
    arrays: dict[str, NDArray[np.float32]] = {}
    array = None
    for band in bands:
        with rasterio.open(image[band]) as src:
            array = src.read(1).astype(np.float32)
            array[image_mask] = np.nan
            arrays[band] = array.copy()
    del array

    emi = 0.95 + (0.01 * arrays['LAI'])
    #LAI
    lai = arrays['LAI']
    emi[lai > 3] = 0.98
    stefBol = np.full(emi.shape, 5.67e-8)

    rl_up = emi * stefBol * (arrays['T_LST'] ** 4)
    save_to_image(image, cal_bands_dr, meta, rl_up, 'RL_UP')
    del(arrays, emi, lai, rl_up, stefBol)


#INSTANTANEOUS INCOMING SHORT-WAVE RADIATION (Rs_down) [W M-2]
def fexp_radshort_down(
    image: dict[str, str],
    z_alt: str,
    t_air: str,
    ur: str,
    sun_elevation: float,
    time_start: datetime,
    meta: dict,
    cal_bands_dr: str,
) -> None:
    image_mask: NDArray[np.bool_] = np.load(image["MASK"])
    arrays: dict[str, NDArray[np.float32]] = {}
    array = None
    with rasterio.open(z_alt) as src:
        array = src.read(1).astype(np.float32)
        array[image_mask] = np.nan
        arrays["Z_ALT"] = array.copy()
    with rasterio.open(t_air) as src:
        array = src.read(1).astype(np.float32)
        array[image_mask] = np.nan
        arrays["T_AIR"] = array.copy()
    with rasterio.open(ur) as src:
        array = src.read(1).astype(np.float32)
        array[image_mask] = np.nan
        arrays["UR"] = array.copy()
    del array

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
    pres = 101.3 * ((293 - (0.0065 * arrays["Z_ALT"]))/ 293) ** 5.26

    #SATURATION VAPOR PRESSURE (es) [KPA]
    es = 0.6108 * (np.exp( (17.27 * arrays["T_AIR"]) / (arrays["T_AIR"] + 237.3)))
    save_to_image(image, cal_bands_dr, meta, es, "ES")

    #ACTUAL VAPOR PRESSURE (ea) [KPA]
    ea = es * arrays["UR"] / 100
    save_to_image(image, cal_bands_dr, meta, ea, "EA")

    #WATER IN THE ATMOSPHERE [mm]
    #GARRISON AND ADLER (1990)
    W = (0.14 * ea * pres) + 2.1

    #SOLAR ZENITH ANGLE OVER A HORIZONTAL SURFACE
    solar_zenith = 90 - sun_elevation
    degree2radian = 0.01745
    solar_zenith_radians = solar_zenith * degree2radian
    cos_theta = np.cos(solar_zenith_radians)

    #BROAD-BAND ATMOSPHERIC TRANSMISSIVITY (tao_sw)
    #ASCE-EWRI (2005)
    tao_sw = 0.35 + 0.627 * np.exp(((-0.00146 * pres)/(1 * cos_theta)) - (0.075 * (W / cos_theta)**0.4))
    save_to_image(image, cal_bands_dr, meta, tao_sw, "tao_sw")

    #INSTANTANEOUS SHORT-WAVE RADIATION (Rs_down) [W M-2]
    rs_down = gsc * cos_theta * tao_sw * dr
    save_to_image(image, cal_bands_dr, meta, rs_down, "RS_DOWN")

    del(arrays, pres, es, ea, W, tao_sw, rs_down)


    #INSTANTANEOUS INCOMING LONGWAVE RADIATION (Rl_down) [W M-2]
    #ALLEN ET AL (2007)
def fexp_radlong_down(
    image: dict[str, str], n_Ts_cold: float, cal_bands_dr: str, meta: dict
) -> None:
    image_mask: NDArray[np.bool_] = np.load(image["MASK"])
    with rasterio.open(image["TAO_SW"]) as src:
        taosw_array = src.read(1).astype(np.float32)
        taosw_array[image_mask] = np.nan

    log_taosw = np.log(taosw_array)
    rl_down = (0.85 * (- log_taosw) ** 0.09) * 5.67e-8 * (n_Ts_cold ** 4)
    save_to_image(image, cal_bands_dr, meta, rl_down, "RL_DOWN")
    del(taosw_array, log_taosw, rl_down)


    #INSTANTANEOUS NET RADIATON BALANCE (Rn) [W M-2]
def fexp_radbalance(image: dict[str, str], cal_bands_dr: str, meta: dict) -> None:
    image_mask: NDArray[np.bool_] = np.load(image["MASK"])
    bands = ["ALFA", "RS_DOWN", "RL_DOWN", "RL_UP", "E_0"]
    arrays: dict[str, NDArray[np.float32]] = {}
    array = None
    for band in bands:
        with rasterio.open(image[band]) as src:
            array = src.read(1).astype(np.float32)
            array[image_mask] = np.nan
            arrays[band] = array.copy()
    del array

    rn = ((1-arrays["ALFA"]) * arrays["RS_DOWN"]) + arrays["RL_DOWN"] - arrays["RL_UP"] - ((1 - arrays["E_0"]) * arrays["RL_DOWN"])
    save_to_image(image, cal_bands_dr, meta, rn, "RN")
    del(arrays, rn)


    #SOIL HEAT FLUX (G) [W M-2]
    #BASTIAANSSEN (2000)
def fexp_soil_heat(image: dict[str, str], cal_bands_dr: str, meta: dict) -> None:
    image_mask: NDArray[np.bool_] = np.load(image["MASK"])
    arrays: dict[str, NDArray[np.float32]] = {}
    bands = ["RN", "NDVI", "ALFA", "T_LST_DEM"]
    arrays = {}
    array = None
    for band in bands:
        with rasterio.open(image[band]) as src:
            array = src.read(1).astype(np.float32)
            array[image_mask] = np.nan
            arrays[band] = array.copy()
    del array

    G = arrays["RN"] * (arrays["T_LST_DEM"] - 273.15) * ( 0.0038 + (0.0074 * arrays["ALFA"])) *  (1 - 0.98 * (arrays["NDVI"] ** 4))
    save_to_image(image, cal_bands_dr, meta, G, "G")
    del(arrays, G)


    #SENSIBLE HEAT FLUX (H) [W M-2]
def fexp_sensible_heat_flux(
    image: dict[str, str],
    ux: str,
    n_Ts_cold: float,
    d_hot_pixel: dict,
    cal_bands_dr: str,
    meta: dict,
) -> None:
    image_mask: NDArray[np.bool_] = np.load(image["MASK"])
    arrays: dict[str, NDArray[np.float32]] = {}
    bands = ["SAVI", "T_LST_DEM"]
    arrays = {}
    array = None
    for band in bands:
        with rasterio.open(image[band]) as src:
            array = src.read(1).astype(np.float32)
            array[image_mask] = np.nan
            arrays[band] = array.copy()
    del array

    src = rasterio.open(ux)
    array = src.read(1).astype(np.float32)
    array[image_mask] = np.nan
    arrays["UX"] = array.copy()

    #VEGETATION HEIGHTS  [M]
    n_veg_hight = 3

    #WIND SPEED AT HEIGHT Zx [M]
    n_zx = 2

    #BLENDING HEIGHT [M]
    n_hight = 200

    #AIR SPECIFIC HEAT [J kg-1/K-1]
    n_Cp = 1004

    #VON KARMAN'S CONSTANT
    n_K = 0.41

    #TS HOT PIXEL
    n_Ts_hot = d_hot_pixel['temp']
    #G HOT PIXEL
    n_G_hot = d_hot_pixel['G']
    #RN HOT PIXEL
    n_Rn_hot = d_hot_pixel['Rn']

    #MOMENTUM ROUGHNESS LENGHT (ZOM) AT THE WEATHER STATION [M]
    #BRUTSAERT (1982)
    n_zom = n_veg_hight * 0.12

    #FRICTION VELOCITY AT WEATHER STATION [M S-1]
    i_ufric_ws = (n_K * arrays['UX'])/ np.log(n_zx /n_zom)

    #WIND SPEED AT BLENDING HEIGHT AT THE WEATHER STATION [M S-1]
    i_u200 = i_ufric_ws *  (np.log(n_hight/n_zom)/n_K)

    #MOMENTUM ROUGHNESS LENGHT (ZOM) FOR EACH PIXEL [M]
    i_zom = np.exp((5.62 * (arrays['SAVI']))-5.809)
    save_to_image(image, cal_bands_dr, meta, i_zom, 'ZOM')
    del(arrays['SAVI'], arrays['UX'], i_ufric_ws)

    #FRICTION VELOCITY FOR EACH PIXEL  [M S-1]
    i_ufric = (n_K *i_u200) /(np.log(n_hight/n_zom))
    save_to_image(image, cal_bands_dr, meta, i_ufric, 'U_FR')

    #AERODYNAMIC RESISTANCE TO HEAT TRANSPORT (rah) [S M-1]

    #Z1 AND Z2 ARE HEIGHTS [M] ABOVE THE ZERO PLANE DISPLACEMENT
    #OF THE VEGETATION
    z1= 0.1
    z2= 2
    i_rah = (np.log(z2/z1))/(i_ufric*0.41)
    save_to_image(image, cal_bands_dr, meta, i_rah, 'RAH_FIRST')

    # i_rah_first = i_rah

    #AIR DENSITY HOT PIXEL
    n_ro_hot= (-0.0046 * n_Ts_hot) + 2.5538

    #========ITERATIVE PROCESS=========#

    #SENSIBLE HEAT FLUX AT THE HOT PIXEL (H_hot)
    n_H_hot = n_Rn_hot - n_G_hot

    #ITERATIVE VARIABLES
    n= 1
    n_dif= 1
    n_dif_min = 0.1
    list_dif = []
    list_dT_hot = []
    list_rah_hot = []
    list_coef_a = []
    list_coef_b = []

    #NUMBER OF ITERATIVE STEPS: 15
    #CAN BE CHANGED, BUT BE AWARE THAT
    #A MINIMUM NUMBER OF ITERATIVE PROCESSES
    #IS NECESSARY TO ACHIEVE RAH AND H ESTIMATIONS

    #========INIT ITERATION========#
    for n in range(15):
    #AERODYNAMIC RESISTANCE TO HEAT TRANSPORT
    #IN HOT PIXEL        
        n_rah_hot = i_rah[d_hot_pixel['index'][0], d_hot_pixel['index'][1]]

    #NEAR SURFACE TEMPERATURE DIFFERENCE IN HOT PIXEL (dT= Tz1-Tz2)  [K]
    # dThot= Hhot*rah/(ρCp)
        n_dT_hot = (n_H_hot * n_rah_hot) / (n_ro_hot * n_Cp)

    #NEAR SURFACE TEMPERATURE DIFFERENCE IN COLD PIXEL (dT= tZ1-tZ2)
        n_dT_cold = 0
    # dT =  aTs + b
    #ANGULAR COEFFICIENT
        n_coef_a = (n_dT_cold - n_dT_hot) / (n_Ts_cold - n_Ts_hot)

    #LINEAR COEFFICIENT
        n_coef_b = n_dT_hot - (n_coef_a * n_Ts_hot)

    #dT FOR EACH PIXEL [K]
        i_lst_med = arrays['T_LST_DEM']
        i_dT_int = (n_coef_a * i_lst_med) + n_coef_b

    #AIR TEMPERATURE (TA) FOR EACH PIXEL (TA=TS-dT) [K]
        i_Ta = i_lst_med - i_dT_int

    #AIR DENSITY (ro) [KM M-3]
        i_ro = (-0.0046 * i_Ta) + 2.5538

    #SENSIBLE HEAT FLUX (H) FOR EACH PIXEL  [W M-2]
        i_H_int = (i_ro*n_Cp*i_dT_int)/i_rah

    #GET VALUE
        n_H_int = i_H_int[d_hot_pixel['index'][0], d_hot_pixel['index'][1]]

    #MONIN-OBUKHOV LENGTH (L)
    #FOR STABILITY CONDITIONS OF THE ATMOSPHERE IN THE ITERATIVE PROCESS
        i_L_int = -(i_ro*n_Cp*(i_ufric**3)*i_lst_med)/(0.41*9.81*i_H_int)
    #STABILITY CORRECTIONS FOR MOMENTUM AND HEAT TRANSPORT
    #PAULSON (1970)
    #WEBB (1970)
        # img = np.full(i_L_int.shape, 0)

    #STABILITY CORRECTIONS FOR STABLE CONDITIONS
        i_psim_200 = -5*(200/i_L_int)
        i_psih_2 = -5*(2/i_L_int)
        i_psih_01 = -5*(0.1/i_L_int)

    #FOR DIFFERENT HEIGHT
        i_x200 = (1-(16*(200/i_L_int)))**0.25
        i_x2 = (1-(16*(2/i_L_int)))**0.25
        i_x01 = (1-(16*(0.1/i_L_int)))**0.25


    #STABILITY CORRECTIONS FOR UNSTABLE CONDITIONS
        i_psimu_200 = 2*np.log((1+i_x200)/2)+np.log((1+i_x200**2)/2)-2*np.arctan(i_x200)+0.5*pi
        i_psihu_2 = 2*np.log((1+i_x2**2)/2)
        i_psihu_01 = 2*np.log((1+i_x01**2)/2)

    #FOR EACH PIXEL
        i_psim_200 = np.where(i_L_int < 0, i_psimu_200, i_psim_200)
        i_psih_2 = np.where(i_L_int < 0, i_psihu_2, i_psih_2)
        i_psih_01 = np.where(i_L_int < 0, i_psihu_01, i_psih_01)
        i_psim_200 = np.where(i_L_int == 0, 0, i_psim_200)
        i_psih_2 = np.where(i_L_int == 0, 0, i_psih_2)
        i_psih_01 = np.where(i_L_int == 0, 0, i_psih_01)

        if n==1:
            i_psim_200_exp = i_psim_200
            i_psih_2_exp = i_psih_2
            i_psih_01_exp = i_psih_01
            i_L_int_exp = i_L_int
            i_H_int_exp = i_H_int
            i_dT_int_exp = i_dT_int
            i_rah_exp = i_rah

    #CORRECTED VALUE FOR THE FRICTION VELOCITY (i_ufric) [M S-1]
        i_ufric = (i_u200*0.41)/(np.log(n_hight/i_zom)-i_psim_200)

    #CORRECTED VALUE FOR THE AERODYNAMIC RESISTANCE TO THE HEAT TRANSPORT (rah) [S M-1]
        i_rah = (np.log(z2/z1)-i_psih_2+i_psih_01)/(i_ufric*0.41)
        if n==1:
            n_dT_hot_old = n_dT_hot
            n_rah_hot_old = n_rah_hot
            n_dif = 1

        if n > 1:
            n_dT_hot_abs = abs(n_dT_hot)
            n_dT_hot_old_abs = abs(n_dT_hot_old)
            n_rah_hot_abs = abs(n_rah_hot)
            n_rah_hot_old_abs = abs(n_rah_hot_old)
            n_dif= abs(n_dT_hot_abs - n_dT_hot_old_abs + n_rah_hot_abs - n_rah_hot_old_abs)
            n_dT_hot_old = n_dT_hot
            n_rah_hot_old = n_rah_hot
        del(i_psihu_01, i_psihu_2, i_psimu_200, i_psih_01, i_psih_2, i_psim_200, i_x01, i_x2, i_x200, i_L_int, i_Ta)
        print(f"n = {n} {n_dif} {n_coef_a} {n_coef_b} {n_dT_hot} {n_rah_hot}")
        #INSERT EACH ITERATION VALUE INTO A LIST
        list_dif.append(n_dif)
        list_coef_a.append(n_coef_a)
        list_coef_b.append(n_coef_b)
        list_dT_hot.append(n_dT_hot)
        list_rah_hot.append(n_rah_hot)

    #=========END ITERATION =========#
    save_to_image(image, cal_bands_dr, meta, i_ufric, 'ufric_star')

    #GET FINAL rah, dT AND H
    i_rah_final = i_rah #[SM-1]
    save_to_image(image, cal_bands_dr, meta, i_rah_final, 'RAH')

    i_dT_final = i_dT_int #[K]
    save_to_image(image, cal_bands_dr, meta, i_dT_final, 'dT')

    i_H_final = (i_ro*n_Cp*i_dT_final)/i_rah_final  #[W M-2]
    save_to_image(image, cal_bands_dr, meta, i_H_final, 'H')

    del(i_rah_final, i_dT_final, i_H_final, i_rah, i_ufric, n_H_int, i_H_int, i_ro, i_dT_int, i_lst_med)


# coregister strm data
def co_dem(
    meta: dict, res: tuple[float, float], srtm_elevation: str, ls_data_dr: str
) -> str:
    with rasterio.open(srtm_elevation) as src:
        dem, _ = reproject(src.read(1), destination= np.zeros([meta["height"], meta["width"]]), src_transform=src.transform,
            src_crs=src.crs, src_nodata=-9999, dst_transform=meta["transform"],
            dst_crs=meta["crs"], dst_nodata=-9999, dst_resolution=res, resampling=Resampling.bilinear)
    strm_file = join(ls_data_dr,"strm_30m_co.tif")
    if exists(strm_file):
        remove(strm_file)
    meta.update(
        nodata= -9999,
        dtype= "int16"
    )
    strm_file = save_data(ls_data_dr, meta, dem, "strm_30m_co")
    del(dem)
    return strm_file

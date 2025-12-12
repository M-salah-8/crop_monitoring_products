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
import numpy as np
from numpy.typing import NDArray

#A SIMPLIFIED VERSION OF
#CALIBRATION USING INVERSE MODELING AT EXTREME CONDITIONS (CIMEC)
#FROM ALLEN ET AL. (2013) FOR METRIC
#SEE MORE: LAIPELT ET AL. (2020)

#DEFAULT PARAMETERS
#NDVI COLD = 5%
#TS COLD = 20%
#NDVI HOT = 10%
#TS HOT = 20%

# SELECT COLD PIXEL
def fexp_cold_pixel(
    pos_ndvi: NDArray[np.float32],
    lst_nw: NDArray[np.float32],
    ndvi: NDArray[np.float32],
    p_top_NDVI: int,
    p_coldest_Ts: int,
) -> tuple[dict, float]:

    ndvi_neg =  pos_ndvi * -1

    # IDENTIFY THE TOP % NDVI PIXELS
    n_perc_top_NDVI= np.nanpercentile(ndvi_neg, p_top_NDVI)

    i_top_NDVI = ndvi_neg
    i_top_NDVI[i_top_NDVI > n_perc_top_NDVI] = np.nan
    lower, upper = np.nanpercentile(i_top_NDVI, 40), np.nanpercentile(i_top_NDVI, 60)
    i_top_NDVI[i_top_NDVI < lower] = np.nan
    i_top_NDVI[i_top_NDVI > upper] = np.nan

    #SELECT THE COLDEST TS FROM PREVIOUS NDVI GROUP
    lst_top_NDVI= np.where(np.isnan(i_top_NDVI), np.nan, lst_nw)
    n_perc_low_LST= np.nanpercentile(lst_top_NDVI, p_coldest_Ts)

    #UPDATE MASK WITH LST VALUES
    i_cold_lst= lst_top_NDVI.copy()
    i_cold_lst[i_cold_lst > n_perc_low_LST] = np.nan

    # FILTERS    ### ??
    c_lst_cold20 = i_cold_lst.copy()
    c_lst_cold20[c_lst_cold20 < 200] = np.nan
    c_lst_cold20_int = np.round(c_lst_cold20)

    lower, upper = np.nanpercentile(c_lst_cold20, 40), np.nanpercentile(c_lst_cold20, 60)
    c_lst_cold20[c_lst_cold20 < lower] = np.nan
    c_lst_cold20[c_lst_cold20 > upper] = np.nan

    #COUNT NUNMBER OF PIXELS
    n_count_final_cold_pix = np.count_nonzero(~np.isnan(c_lst_cold20))

    #SELECT COLD PIXEL RANDOMLY (FROM PREVIOUS SELECTION)
    non_nan_indices = np.where(~np.isnan(c_lst_cold20))
    index = np.random.choice(len(non_nan_indices[0]))
    i_0, i_1 = non_nan_indices[0][index], non_nan_indices[1][index]

    n_Ts_cold: float = float(lst_nw[i_0, i_1])
    # n_long_cold = ee.Number(fc_cold_pix.aggregate_first('longitude'))
    # n_lat_cold = ee.Number(fc_cold_pix.aggregate_first('latitude'))
    n_ndvi_cold = ndvi[i_0, i_1]

    d_cold_pixel = {
        "ndvi": n_ndvi_cold,
        "index": [i_0, i_1],
        "sum": n_count_final_cold_pix,
    }

    return d_cold_pixel, n_Ts_cold


#SELECT HOT PIXEL
def fexp_hot_pixel(
    pos_ndvi: NDArray[np.float32],
    lst: NDArray[np.float32],
    G: NDArray[np.float32],
    rn: NDArray[np.float32],
    ndvi: NDArray[np.float32],
    lst_nw: NDArray[np.float32],
    p_lowest_NDVI: int,
    p_hottest_Ts: int,
) -> dict:
    #IDENTIFY THE DOWN % NDVI PIXELS
    n_perc_low_NDVI= np.nanpercentile(pos_ndvi, p_lowest_NDVI)

    #UPDATE MASK WITH NDVI VALUES
    i_low_NDVI= pos_ndvi.copy()
    i_low_NDVI[i_low_NDVI > n_perc_low_NDVI] = np.nan
    lower_ndvi = np.nanpercentile(i_low_NDVI, 95)
    i_low_NDVI[i_low_NDVI < lower_ndvi] = np.nan
    #SELECT THE HOTTEST TS FROM PREVIOUS NDVI GROUP
    lst_neg = lst * -1
    lst_low_NDVI= np.where(np.isnan(i_low_NDVI), np.nan, lst_neg)
    n_perc_top_lst= np.nanpercentile(lst_low_NDVI, p_hottest_Ts)

    c_lst_hotpix= lst_low_NDVI.copy()
    c_lst_hotpix[c_lst_hotpix > n_perc_top_lst] = np.nan
    c_lst_hotpix_int=np.round(c_lst_hotpix)

    lower, upper = np.nanpercentile(c_lst_hotpix, 40), np.nanpercentile(c_lst_hotpix, 60)
    c_lst_hotpix[c_lst_hotpix < lower] = np.nan
    c_lst_hotpix[c_lst_hotpix > upper] = np.nan

    #COUNT NUNMBER OF PIXELS
    n_count_final_hot_pix = np.count_nonzero(~np.isnan(c_lst_hotpix))

    #SELECT HOT PIXEL RANDOMLY (FROM PREVIOUS SELECTION)
    non_nan_indices = np.where(~np.isnan(c_lst_hotpix_int))
    index = np.random.choice(len(non_nan_indices[0]))
    i_0, i_1 = non_nan_indices[0][index], non_nan_indices[1][index]

    n_Ts_hot = lst_nw[i_0, i_1]
    # n_long_hot = ee.Number(fc_hot_pix.aggregate_first("longitude"))
    # n_lat_hot = ee.Number(fc_hot_pix.aggregate_first("latitude"))  n_ndvi_hot = ndvi[i_0, i_1]
    n_ndvi_hot = ndvi[i_0, i_1]
    n_Rn_hot = rn[i_0, i_1]
    n_G_hot = G[i_0, i_1]
    #CREATE A DICTIONARY WITH THOSE RESULTS
    d_hot_pixel = {
        "temp": n_Ts_hot,
        "index": [i_0, i_1],
        "Rn": n_Rn_hot,
        "G": n_G_hot,
        "ndvi": n_ndvi_hot,
        "sum": n_count_final_hot_pix,
    }

    return d_hot_pixel


def hot_cold_pixels_helpers(
    ndvi: NDArray[np.float32], ndwi: NDArray[np.float32], lst: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    pos_ndvi = ndvi.copy()
    pos_ndvi[pos_ndvi <= 0] = np.nan
    # ndvi_is_valid = np.full(ndvi.shape, 1, np.float32)
    # ndvi_is_valid[np.isnan(ndvi)] = np.nan
    # sd_ndvi = np.full(ndvi.shape, 1, np.float32)
    # sd_ndvi[np.isnan(ndvi)] = np.nan
    # TODO: use lst_dem ??
    lst_nw = lst.copy()
    lst_nw[ndwi > 0] = np.nan

    return pos_ndvi, lst_nw


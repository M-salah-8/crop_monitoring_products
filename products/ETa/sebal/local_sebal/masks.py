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

from util.logs import logger


# FUNCTION FO MASK CLOUD IN LANDSAT 8 FOR SURFACE REFELCTANCE    ### reduce none values
def _f_cloudMaskL_8_9_SR(pixel_qa: NDArray[np.uint16]) -> NDArray[np.bool_]:
    """Returns a cloud mask"""
    cloud_mask = (pixel_qa & ((1 << 1) | (1 << 3) | (1 << 4))) != 0
    return cloud_mask


def f_final_mask(
    pixel_qa: NDArray[np.uint16], blue: NDArray[np.float32], nodata: float | int
) -> NDArray[np.bool_]:
    """Returns a mask that contains nodata and cloud locations"""
    cloud_mask = _f_cloudMaskL_8_9_SR(pixel_qa)
    if np.isnan(nodata):
        nodata_mask = np.isnan(blue)
    else:
        nodata_mask = blue == nodata

    mask = cloud_mask | nodata_mask
    return mask


#ALBEDO
#USING TASUMI ET AL. (2008) METHOD FOR LANDSAT 8
# COEFFICIENTS FROM KE ET AL. (2016)
def f_albedoL_8_9(
b_ca, b_blue, b_green, b_red, b_nir, b_swir_1, b_swir_2
) -> NDArray[np.float32]:

    logger.info("calculate albedo")

    alfa = (
        (0.130 * b_ca)
        + (0.115 * b_blue)
        + (0.143 * b_green)
        + (0.180 * b_red)
        + (0.281 * b_nir)
        + (0.108 * b_swir_1)
        + (0.042 * b_swir_2)
    )
    return alfa

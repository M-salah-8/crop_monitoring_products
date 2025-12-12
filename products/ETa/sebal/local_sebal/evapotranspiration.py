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
#Call EE
# import ee
from os import makedirs
from os.path import join

import numpy as np
from numpy.typing import NDArray

from .funcs import save_data


def fexp_et(
    rn: NDArray[np.float32],
    G: NDArray[np.float32],
    lst_dem: NDArray[np.float32],
    H: NDArray[np.float32],
    rn24hobs: NDArray[np.float32],
    meta: dict,
    results_dr: str,
    image_date: str,
) -> None:
    #NET DAILY RADIATION (Rn24h) [W M-2]
    #BRUIN (1982)
    # rn24hobs_array = rn24hobs_array * 1

    # GET ENERGY FLUXES VARIABLES AND LST
    i_Rn = rn
    i_G = G
    i_lst = lst_dem
    i_H_final = H

    #FILTER VALUES
    i_H_final[i_H_final < 0]= 0

    # INSTANTANEOUS LATENT HEAT FLUX (LE) [W M-2]
    #BASTIAANSSEN ET AL. (1998)
    i_lambda_ET = i_Rn-i_G-i_H_final
    #FILTER
    i_lambda_E= np.where(i_lambda_ET < 0, 0, i_lambda_ET)
    del(i_lambda_E)

    #LATENT HEAT OF VAPORIZATION (LAMBDA) [J KG-1]
    #BISHT ET AL.(2005)
    #LAGOUARDE AND BURNET (1983)
    i_lambda = 2.501-0.002361*(i_lst-273.15)

    #INSTANTANEOUS ET (ET_inst) [MM H-1]
    i_ET_inst = 0.0036 * (i_lambda_ET/i_lambda)

    #EVAPORATIVE FRACTION (EF)
    #CRAGO (1996)
    i_EF = i_lambda_ET/(i_Rn-i_G)

    #DAILY EVAPOTRANSPIRATION (ET_24h) [MM DAY-1]
    i_ET24h_calc = (0.0864 *i_EF * rn24hobs)/(i_lambda)
    output_dr = join(results_dr, "eta")
    makedirs(output_dr, exist_ok=True)
    save_data(output_dr, meta, i_ET24h_calc, image_date)

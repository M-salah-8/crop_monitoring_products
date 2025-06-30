import glob
import os
import funcs.GIS_function as gis

def process_biomass(AOT, fc, MC, tif_template, resample_template, dekads_dr, WaPOR_dr):
  driver, NDV, xsize, ysize, GeoT, Projection = gis.GetGeoInfo(tif_template)

  wapor_npp = sorted(glob.glob(os.path.join(WaPOR_dr, "L2-NPP-D")+'/*.tif'))
  for npp_tif in wapor_npp:
    dekad = os.path.basename(npp_tif).split('.')[0].split('_')[-1]
    NPP  = gis.OpenAsArray(npp_tif, nan_values=True)
    AGBM = (AOT * fc * (NPP * 22.222 / (1 - MC))) / 1000  # Above ground biomass, 1000 is to covert from kg to ton
    os.makedirs(os.path.join(WaPOR_dr, "L2-biomass-D"), exist_ok=True)
    biomass_dr    = os.path.join(WaPOR_dr, "L2-biomass-D", f"{dekad}.tif")
    gis.CreateGeoTiff(biomass_dr, AGBM, driver, NDV, xsize, ysize, GeoT, Projection)
    
  wapor_biomass = sorted(glob.glob(os.path.join(WaPOR_dr, "L2-biomass-D")+'/*.tif'))
  for biomass_tif in wapor_biomass:
      dekad = os.path.basename(biomass_tif).split('.')[0].split('_')[-1]
      resample = gis.MatchProjResNDV(
        resample_template, [os.path.join(WaPOR_dr, "L2-biomass-D", biomass_tif)], os.path.join(dekads_dr, dekad, "biomass"),
        resample = 'near', dtype = 'float32')
      os.rename(resample[0],
        resample[0].replace(
          os.path.basename(resample[0]),
          dekad + ".tif"
          ))
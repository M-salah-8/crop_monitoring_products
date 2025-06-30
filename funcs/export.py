import ee
import requests
import os
import glob
import rasterio
import numpy as np
import pandas as pd
import datetime
import rasterio.mask
from wapordl import wapor_map
from .geo import GeoFunctions as gis
from .time_periods import date_month, date_dekad, date_5days


def export_gee_tifs_localy(image, data, data_name, output_folder, date, aoi):
    data_url = {
        'NDVI': image.select('NDVI').getDownloadUrl({
            'bands': 'NDVI',
            'scale': 30,
            'region': aoi,
            'format': 'GEO_TIFF'
            }),
        'ET_24h': image.select('ET_24h').getDownloadUrl({
            'bands': 'ET_24h',
            'scale': 30,
            'region': aoi,
            'format': 'GEO_TIFF'
            }),
        'RGB': image.select(['R','GR','B']).multiply(0.0000275).add(-0.2).getDownloadUrl({
            'bands': ['R','GR','B'],
            'scale': 30,
            'region': aoi,
            'format': 'GEO_TIFF'
            })
    }
    response = requests.get(data_url[data])
    os.makedirs(os.path.join(output_folder, data_name), exist_ok= True)
    with open(os.path.join(output_folder, data_name, f'{data_name}_{date}.tif'), 'wb') as fd:
      fd.write(response.content)

def export_image_to_drive(img, description, folder, aoi):
    # Export cloud-optimized GeoTIFF images
    ee.batch.Export.image.toDrive(**{
        'image': img,
        'description': description,
        'scale': 30,
        "folder": folder,
        'region': aoi,
        'fileFormat': 'GeoTIFF',
        'maxPixels': 3784216672400,
        'formatOptions': {
            'cloudOptimized': True
        }
    }).start()

def export_to_drive(result_img, date, folder, aoi):  ### fix date (but outside fun)
    export_image_to_drive(
        result_img.image.select(['R','GR','B']).multiply(0.0000275).add(-0.2),  ### check
        "rgb_" + date,
        folder,
        aoi
    )
    export_image_to_drive(
        result_img.image.select('NDVI'),
        "ndvi_" + date,
        folder,
        aoi
    )
    export_image_to_drive(
        result_img.image.select('ET_24h'),
        "eta_" + date,
        folder,
        aoi
    )

def download_WaPOR(region, variables, period, season_dr, overview = "NONE"):
  for var in variables:
    download_dr = os.path.join(season_dr, 'WaPOR_data', var)
    os.makedirs(download_dr, exist_ok=True)

    if('-E' in var):
      unit = "day"
    elif('-D' in var):
      unit = "dekad"
    elif('-M' in var):
      unit = "month"
    elif ('-A' in var):
      unit = "year"
    else:
      unit = "none"

    wapor_map(region, var, period, download_dr, seperate_unscale = True, unit_conversion = unit)
        
def export_local_tifs(data_drs, season_dr, project_gdf):
  for data_dr in data_drs:
    tifs_drs = glob.glob(data_dr+'/*.tif')
    for tif in tifs_drs:
      date = os.path.basename(tif).split(".")[0].split("_")[1]
      data_name = os.path.basename(tif).split(".")[0].split("_")[0]
      date_d = date_dekad(date)
      src = rasterio.open(tif)
      pet_array, transform = rasterio.mask.mask(src, project_gdf.to_crs(src.crs).geometry, crop = True)
      meta = src.meta
      meta.update(width= pet_array.shape[-1], height= pet_array.shape[-2], transform= transform)
      download_dr = os.path.join(season_dr, "dekads", date_d, data_name, 'tifs')
      os.makedirs(download_dr, exist_ok=True)
      with rasterio.open(os.path.join(download_dr, f'{data_name}_{date}.tif'), 'w', **meta) as dst:
        dst.write(pet_array)

def results_to_5days(tifs_dir, output_dir, data_name, template, resampling):
  gis_in = gis()
  for tif_dir in tifs_dir:
    file_name = os.path.basename(tif_dir).split(".")[0]
    date = file_name.split("_")[1]
    dst_date = date_5days(date)
    out_dir = os.path.join(output_dir, dst_date, data_name)
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"{file_name}.tif")
    if not os.path.exists(dst):
      array, meta = gis_in.coregister(template, tif_dir, dtype=np.float32, resampling= resampling)
      gis_in.save_tif(array, meta, file_name, out_dir)
    else:
      print(f'{file_name} already exists')

def results_to_dekad(tifs_dir, output_dir, data_name, template, resampling):
  gis_in = gis()
  for tif_dir in tifs_dir:
    file_name = os.path.basename(tif_dir).split(".")[0]
    date = file_name.split("_")[1]
    dst_date = date_dekad(date)
    out_dir = os.path.join(output_dir, dst_date, data_name)
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"{file_name}.tif")
    if not os.path.exists(dst):
      array, meta = gis_in.coregister(template, tif_dir, dtype=np.float32, resampling= resampling)
      gis_in.save_tif(array, meta, file_name, out_dir)
    else:
      print(f'{file_name} already exists')

def results_to_month(tifs_dir, output_dir, data_name, template, resampling):
  gis_in = gis()
  for tif_dir in tifs_dir:
    file_name = os.path.basename(tif_dir).split(".")[0]
    date = file_name.split("_")[1]
    dst_date = date_month(date)
    out_dir = os.path.join(output_dir, dst_date, data_name)
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"{file_name}.tif")
    if not os.path.exists(dst):
      array, meta = gis_in.coregister(template, tif_dir, dtype=np.float32, resampling= resampling)
      gis_in.save_tif(array, meta, file_name, out_dir)
    else:
      print(f'{file_name} already exists')

def product_to_results(tifs_dir, output_dir, data_name, time_period, template, resampling = "nearest"):
  output_dir = os.path.join(output_dir, time_period)
  if time_period == '5days':
    results_to_5days(tifs_dir, output_dir, data_name, template, resampling)
  elif time_period == 'dekad':
    results_to_dekad(tifs_dir, output_dir, data_name, template, resampling)
  elif time_period == 'month':
    results_to_month(tifs_dir, output_dir, data_name, template, resampling)
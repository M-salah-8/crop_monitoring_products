# Disclaimer
* This code is for educational purposes and should not be used in production environments as it is untested.

# new version
* The project is now "Crop Monitoring" (will rename the repo later)
* Currently, the project consists of a few products, but it will grow in the future.
* Analysis and visualisation will be improved upon and added again later.

# products
## [geeSEBAL](docs/sebal.md)
Surface Energy Balance Algorithm for Land (SEBAL), a model for estimating evapotranspiration (ET) from energy balance equation.\
geeSEBAL is an open-source implementation of (SEBAL) using Google Earth Engine (GEE).\
outputs:
* ETa = actual evapotranspiration
* NDVI = Normalized Difference Vegetation Index
* RGB = red,green,blue image

## [SEBAL](docs/sebal.md)
Surface Energy Balance Algorithm for Land (SEBAL), a model for estimating evapotranspiration (ET) from energy balance equation.\
outputs:
* ETa = actual evapotranspiration
* NDVI = Normalized Difference Vegetation Index
* RGB = red,green,blue image

## [growth](docs/growth.md)
Using Sentinel Application Platform (SNAP) command line feature to estimate leaf area index (LAI).
outputs:
* LAI = leaf area index
* NDVI = Normalized Difference Vegetation Index

## [WaPOR data](https://www.fao.org/in-action/remote-sensing-for-water-productivity/en)
FAO’s portal to monitor Water Productivity through Open access of Remotely (WaPOR) sensed derived data, monitors and reports on agriculture water productivity over Africa and the Near East.

# Folders layout
```bash
┌─── project_folder ───┐
│   ├── area
│   │   ├── project.geojson
│   │   ├── project_field.shp
│   │   └── project_RET.geojson
│   ├── local_data
│   │   ├── landsat
│   │   │   └── images
|   │   │       ├── 2024-06-17
|   │   │       |   └── C09_L2SP_173048_20240617_20240618_02_T1
|   │   │       ├── 2024-07-11
|   │   │       |   └── C08_L2SP_173048_20240711_20240719_02_T1
|   │   │      ...
|   │   │      
│   │   ├── sentinel_2
|   │   │   ├── 2023_11_04
|   │   │   |   └── S2B_MSIL2A_20231104T083029_N0509_R021_T35QRA_20231104T101901.SAFE.zip
|   │   │   ├── 2023_11_14
|   │   │   |   └── S2B_MSIL2A_20231114T083119_N0509_R021_T35QRA_20231114T102315.SAFE.zip
|   │   │  ...
|   │   │
|   │   └── strm_30m.tif
|   │   
│  ...
│   └── README.md
└─────────────────────┘
```

## [SEN-ET](docs/set-et.md)
* New.
* Not documented yet.

### contact:
Mohammed Salah (mohammedalmak98@gmail.com)  

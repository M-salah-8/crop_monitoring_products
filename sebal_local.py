from os.path import dirname, join, isdir
from glob import glob

from products.ETa.sebal.local_sebal.image import sebal_local

season = "2023"

working_dir = dirname(__file__)
output_dir = join(working_dir, "output", "products_outputs", season, "SEBAL")

data_dir = join(working_dir, "data")
local_ls_images = sorted([
    dir for dir in glob(join(data_dir, "landsat", "images", "*", "*"))
    if isdir(dir)
])

for local_image in local_ls_images:
    sebal_local(
        local_image,
        data_dir,
        output_dir,
        p_top_NDVI = 1,
        p_coldest_Ts = 1,
        p_lowest_NDVI = 1,
        p_hottest_Ts = 1,
    )

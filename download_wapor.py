from os.path import dirname, join

from products.wapor import download_WaPOR


season = "2023"
region = "/home/m-salah/main/work/crop_monitoring/crop_monitoring_code/data/areas/etp_sudan.geojson"

working_dir = dirname(__file__)
output_dir = join(working_dir, "output", "products_outputs", season, "wapor")

# period = [start_date, end_date]
period = ["2023-12-01", "2024-02-10"]
overview = "NONE"
# for all variables https://colab.research.google.com/github/un-fao/FAO-Water-Applications/blob/main/WaPOR/WaPORv3_API.ipynb#scrollTo=CwwxvIZdIiHQ
variables = ["L1-RET-E"]


download_WaPOR(region, variables, period, output_dir, overview)

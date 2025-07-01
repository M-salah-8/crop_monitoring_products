from os.path import join

from numpy.typing import NDArray
import rasterio


def save_data(output_dir: str, meta: dict, array: NDArray, name: str) -> str:
    tif_file = join(output_dir, f"{name.lower()}.tif")
    with rasterio.open(tif_file, "w", **meta) as dst:
        dst.write(array, 1)
    return tif_file

def save_to_image(image: dict[str, str], output_dir: str, meta: dict, array: NDArray, name: str) -> None:
    image[name.upper()] = save_data(output_dir, meta, array, name)






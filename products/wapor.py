from os import makedirs, rename
from os.path import basename, join
from shutil import rmtree
from glob import glob

from wapordl import wapor_map


def download_WaPOR(
    region: list[float] | str,
    variables: list[str],
    period: list[str],
    output_dir: str,
    overview: str = "NONE",
) -> None:
    for var in variables:
        download_dir = join(output_dir, f"{var}_download")
        final_dir = join(output_dir, var)
        makedirs(download_dir, exist_ok=True)
        makedirs(final_dir, exist_ok=True)

        if "-E" in var:
            unit = "day"
        elif "-D" in var:
            unit = "dekad"
        elif "-M" in var:
            unit = "month"
        elif "-A" in var:
            unit = "year"
        else:
            unit = "none"

        try:
            wapor_map(
                region,
                var,
                period,
                download_dir,
                unit_conversion=unit,
                overview=overview,
                separate_unscale=True,
            )
            downloaded_files = glob(join(download_dir, "*.tif"))
            for file in downloaded_files:
                new_tif = basename(file).split("_")[-1]    # yyyy-mm-dd.tif
                rename(file, join(final_dir, new_tif))
        finally:
            rmtree(download_dir)

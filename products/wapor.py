from os import makedirs
from os.path import join

from wapordl import wapor_map


def download_WaPOR(
    region: list[float] | str,
    variables: list[str],
    period: list[str],
    output_dr: str,
    overview: str = "NONE",
) -> None:
    for var in variables:
        download_dr = join(output_dr, var)
        makedirs(download_dr, exist_ok=True)

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

        wapor_map(
            region,
            var,
            period,
            download_dr,
            unit_conversion=unit,
            overview=overview,
            separate_unscale=True,
        )

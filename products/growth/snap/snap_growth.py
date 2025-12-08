from os import makedirs
from os.path import dirname, join, exists, basename
import subprocess
from enum import Enum
from xml.etree import ElementTree as ET


class SnapProducts(Enum):
    lai = "lai.xml"
    cab = "cab.xml"
    ndvi = "ndvi.xml"


def write_xml(
    input_xml: str, output_xml: str, input_file: str, output_file: str
) -> None:
    tree = ET.parse(input_xml)
    root = tree.getroot()
    nodes = root.findall("node")
    files = [
        {"id": "Read", "file": input_file},
        {"id": "Write", "file": output_file}
    ]
    for file in files:
        node_element = next((node for node in nodes if node.get("id") == file["id"]), None)
        assert node_element is not None
        file_element = node_element.find(".//file")
        assert file_element is not None
        file_element.text = file["file"]
    tree.write(output_xml)


def snap_processes(product: SnapProducts, gpt: str, image_file: str, output_dir: str):
    input_xml = join(dirname(__file__), product.value)
    assert exists(input_xml), f"graph does not exist: {input_xml}"
    product_output_dir = join(output_dir, product.name)
    makedirs(product_output_dir, exist_ok=True)

    image_date = basename(image_file).split('_')[-1].split('.')[0][0:8]
    image_date = f"{image_date[0:4]}-{image_date[4:6]}-{image_date[6:8]}"
    output_tif = join(product_output_dir, f"{image_date}.tif")

    graphs_dir = join(product_output_dir, "graphs")
    makedirs(graphs_dir, exist_ok=True)
    output_xml = join(graphs_dir, f"{image_date}.xml")
    write_xml(input_xml, output_xml, image_file, output_tif)

    subprocess.run([gpt, output_xml])

from sentinelhub import (
    SHConfig,
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)

import datetime
import os
from PIL import Image
import numpy as np

#import matplotlib.pyplot as plt
import numpy as np

class satellite_img_requester:

    evalscript_true_color = """
        // VERSION=3

        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04"]  // RGB bands
                }],
                output: {
                    bands: 3
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02];  // return RGB bands
        }
    """

    def __init__(self, clientID, client_secret):
        self.config = SHConfig()
        self.config.sh_client_id = clientID
        self.config.sh_client_secret = client_secret
        self.config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        self.config.sh_base_url = "https://sh.dataspace.copernicus.eu"

    def set_bounding_box(self,
                         left_bottom_L,
                         left_bottom_B,
                         right_top_L,
                         right_top_B
                         ):
        coords_wgs84 = (left_bottom_L, left_bottom_B, right_top_L, right_top_B)
        self.bbox = BBox(bbox=coords_wgs84,
                         crs=CRS.WGS84 # 'EPSG:4326'
                         )

    def set_resolution(self, m_per_pix):
        self.resolution = m_per_pix

    def set_time_interval(self, start_date, end_date):
        self.time_interval = (start_date, end_date)

    def get_image(self):
        self.size = bbox_to_dimensions(self.bbox, resolution=self.resolution)

        request_true_color = SentinelHubRequest(
            evalscript=self.evalscript_true_color,
            input_data=[SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C.define_from("s2l1c", service_url=self.config.sh_base_url),
                time_interval=self.time_interval,
                mosaicking_order=MosaickingOrder.LEAST_CC,
            )],
            responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],  # output PNG
            bbox=self.bbox,  # set bbox
            size=self.size,
            config=self.config,
        )
        true_color_imgs = request_true_color.get_data()
        self.image = true_color_imgs[0]
        return self.image, self.bbox

    def save_image(self, path):
        Image.fromarray(self.image).save(os.path.join(path, "test.png"), format="PNG")

if __name__ == "__main__":
    ID = "sh-98359940-55d0-4966-82f5-a6bc72c45edc"
    secret = "22vAcEHSRAQrvHkVJeYHyDi4lWOVhut4"
    downloader = satellite_img_requester(ID, secret)
    downloader.set_bounding_box(112.924866,28.155289,112.933921,28.163556)
    downloader.set_resolution(10)
    downloader.set_time_interval("2024-06-12", "2025-06-12")
    downloader.get_image()
    downloader.save_image("./")
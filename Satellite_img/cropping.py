import numpy as np
from scipy.ndimage import zoom
from rasterio.transform import from_origin
import rasterio
import os
from Satellite_img.histotools import process
from rasterio.transform import array_bounds
from PIL import Image

def split_bbox(bbox, n):
    min_L, min_B, max_L, max_B = bbox
    delta_L = (max_L - min_L) / n
    delta_B = (max_B - min_B) / n

    grid = []
    for i in range(n):
        for j in range(n):
            sub_bbox = [
                min_L + i * delta_L,
                min_B + j * delta_B,
                min_L + (i + 1) * delta_L,
                min_B + (j + 1) * delta_B
            ]
            grid.append(sub_bbox)

    return grid

class GeoTIFF:
    def __init__(self,
                 img: np.ndarray,
                 bbox: list # [min L, min_B, max_L, max_B]
                 ):
        self.img = img
        self.bbox = bbox

class Cropping:
    def __init__(self):
        self.img_list = []

    def crop_image_array(self,
                         img: np.ndarray,
                         bbox: list, # [min L, min_B, max_L, max_B]
                         crop_height: int,
                         crop_width: int,
                         ):
        self.img_list = []
        min_L, min_B, max_L, max_B = bbox

        img_height, img_width = img.shape[:2]
        new_height = int((img_height // crop_height) * crop_height)
        new_width = int((img_width // crop_width) * crop_width)
        # height and width of the downsampled image

        scale_height = new_height / img_height
        scale_width = new_width / img_width
        if len(img.shape) == 2:
            img = zoom(img, (scale_height, scale_width), order=3)
        elif len(img.shape) == 3:
            img = zoom(img, (scale_height, scale_width, 1), order=3)
            # downsampled image

        B_range = max_B - min_B
        L_range = max_L - min_L

        for i in range(0, new_height, crop_height):
            for j in range(0, new_width, crop_width):
                crop_min_B = max_B - ((i + crop_height) / new_height) * B_range
                crop_max_B = max_B - (i / new_height) * B_range
                crop_min_L = min_L + (j / new_width) * L_range
                crop_max_L = min_L + ((j + crop_width) / new_width) * L_range

                cropped_img = img[int(i):int(i+crop_height), int(j):int(j+crop_width)]
                cropped_img = process(cropped_img)

                self.img_list.append(GeoTIFF(cropped_img, [crop_min_L, crop_min_B, crop_max_L, crop_max_B]))

    def crop_image_file(self, tiff_file: str, crop_height: int, crop_width: int):
        self.img_list = []
        with rasterio.open(tiff_file) as src:
            transform = src.transform
            height, width = src.height, src.width
            bands = src.count-1
            min_L, min_B, max_L, max_B = array_bounds(
                height, width, transform
            )
            bbox = [min_L, min_B, max_L, max_B]

            new_height = int((height // crop_height) * crop_height)
            new_width = int((width // crop_width) * crop_width)
            # height and width of the downsampled image
            img = src.read()[:3]
            img_ds = np.zeros((bands, new_height, new_width), dtype=img.dtype)

            for c in range(bands):
                img_c = Image.fromarray(img[c])
                img_c = img_c.resize((new_width, new_height), Image.BILINEAR)
                img_ds[c] = np.array(img_c)

            B_range = max_B - min_B
            L_range = max_L - min_L
            
            for i in range(0, new_height, crop_height):
                for j in range(0, new_width, crop_width):
                    crop_min_B = max_B - ((i + crop_height) / new_height) * B_range
                    crop_max_B = max_B - (i / new_height) * B_range
                    crop_min_L = min_L + (j / new_width) * L_range
                    crop_max_L = min_L + ((j + crop_width) / new_width) * L_range

                    cropped_img = img_ds[:, int(i):int(i+crop_height), int(j):int(j+crop_width)]
                    cropped_img = np.transpose(cropped_img, (1, 2, 0))
                    #cropped_img = process(cropped_img)
                    self.img_list.append(GeoTIFF(cropped_img,
                                                 [crop_min_L, crop_min_B, crop_max_L, crop_max_B]
                                                 )
                                                 )

    def save_cropped_images(self, output_dir: str):
        for i, img in enumerate(self.img_list):
            height, width = img.img.shape[0:2]
            min_L, min_B, max_L, max_B = img.bbox

            pixel_B = (max_B - min_B) / height
            pixel_L = (max_L - min_L) / width

            transform = from_origin(min_L, max_B, pixel_L, pixel_B)
            count = 1 if len(img.img.shape) == 2 else img.img.shape[2]

            metadata = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': count,
                'dtype': img.img.dtype,
                'transform': transform,
                'crs': 'EPSG:4326'
            }

            filename = os.path.join(output_dir, f'{min_L}_{max_B}.tif')
            with rasterio.open(filename, 'w', **metadata) as dst:
                if count == 1:
                    dst.write(img.img, 1)
                else:
                    for c in range(count):
                        dst.write(img.img[:, :, c], c+1)
            print(f'Saved {filename}')
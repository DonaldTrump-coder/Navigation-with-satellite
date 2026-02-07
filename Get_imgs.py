from Satellite_img.img_requester import satellite_img_requester
from Satellite_img.cropping import Cropping, split_bbox
from pathlib import Path

if __name__ == "__main__":
    ID = "sh-98359940-55d0-4966-82f5-a6bc72c45edc"
    secret = "22vAcEHSRAQrvHkVJeYHyDi4lWOVhut4"
    bbox = [113.804077,22.200629,114.457764,22.839983]
    region = "HongKong"
    start_time = '2024-06-12'
    end_time = '2025-06-12'
    downloader = satellite_img_requester(ID, secret)

    bbox_list = split_bbox(bbox, 8)
    for i, bbox in enumerate(bbox_list):
        downloader.set_bounding_box(bbox[0], bbox[1], bbox[2], bbox[3])
        downloader.set_resolution(4)
        downloader.set_time_interval(start_time, end_time)
        img, bbox = downloader.get_image()
        cropper = Cropping()
        cropper.crop_image_array(
            img = img,
            bbox = bbox,
            crop_height = 640,
            crop_width = 640
        )
        folder_path = Path(f"./data/Sentinel/{region}")
        folder_path.mkdir(parents=True, exist_ok=True)
        cropper.save_cropped_images(folder_path)
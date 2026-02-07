from Satellite_img.cropping import Cropping
from pathlib import Path
import shutil

if __name__ == '__main__':
    tiff_file = 'data/Google/Changsha.tif'
    region = "Changsha"

    crop = Cropping()
    crop.crop_image_file(tiff_file, 640, 640)
    folder_path = Path(f"./data/Google/{region}")
    folder_path.mkdir(parents=True, exist_ok=True)
    if folder_path.exists():
        for item in folder_path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    crop.save_cropped_images(folder_path)
import rasterio

def get_bound(file_path):
    with rasterio.open(file_path) as src:
        bounds = src.bounds
    return bounds

def geo_to_pixel(lon, lat, view_min_lon, view_max_lon, view_min_lat, view_max_lat, height, width):
    x = (lon - view_min_lon) / (view_max_lon - view_min_lon) * width
    y = (view_max_lat - lat) / (view_max_lat - view_min_lat) * height
    return int(x), int(y)
import requests
import base64
import cv2
import numpy as np
import rasterio
import pickle

base_url = "http://localhost:8000" # url of the server
def send_img(tif_path: str):
    with rasterio.open(tif_path) as src:
        img = src.read()
        bounds = src.bounds
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    
    min_lon, min_lat, max_lon, max_lat = (
            bounds.left, bounds.bottom,
            bounds.right, bounds.top
        )
    payload = {
        "img": img_base64,
        "min_lon": float(min_lon),
        "max_lon": float(max_lon),
        "min_lat": float(min_lat),
        "max_lat": float(max_lat),
    }
    url = base_url + "/infer"
    res = requests.post(url, json=payload)
    return res.json()

def get_patches():
    url = base_url + "/get_patches"
    res = requests.get(url)
    data = pickle.loads(res.content)
    patches = data["patches"]
    texts = data["texts"]
    return patches, texts

def send_descriptions(descriptions):
    url = base_url + "/set_descriptions"
    data = pickle.dumps(descriptions)
    response = requests.post(url, data=data)
    return response.json()

def get_scene_graph():
    url = base_url + "/scene_description"
    response = requests.post(url)
    return response.json()["scene_description"]

def get_trajectory(llm_answers, start_point, survey_areas):
    data = {
        "llm_answers": llm_answers,
        "start_point": start_point,
        "survey_areas": survey_areas
    }
    url = base_url + "/get_trajectory"
    response = requests.get(url, json=data)
    data = response.json()["traj_points"]
    traj_points = [Traj_Point(**p) for p in data]
    return traj_points

class Traj_Point:
    def __init__(self, kind, x, y):
        self.kind = kind
        self.x = x
        self.y = y
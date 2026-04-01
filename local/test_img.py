import rasterio
from tools import send_img, get_patches, send_descriptions, get_scene_graph, get_trajectory
from PIL import Image

def main():
    tif_path = "./data/Google/Changsha/112.922586488_28.164847797.tif"
    res = send_img(tif_path)
    print(res)
    patches, texts = get_patches()
    num_patch = len(patches)
    
    img = Image.open(tif_path)
    width, height = img.size
    start_point = (width / 2, height / 2)
    
    descriptions = []
    for i in range(num_patch):
        descriptions.append("1")
    res = send_descriptions(descriptions)
    print(res)
    
    scene_description = get_scene_graph()
    print(scene_description)
    
    llm_answers = """Answer:
    Params: [[10, 1.1, 5, 1]]
    Routes: [[0, 1, 2]]
    """
    survey_areas = [[2]]
    traj = get_trajectory(llm_answers, start_point, survey_areas)
    print(traj)
    
if __name__ == '__main__':
    main()
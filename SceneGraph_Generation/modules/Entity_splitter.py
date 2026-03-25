import numpy as np
from scipy.ndimage import label
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def split_entities(binary_map: np.ndarray, min_ratio = 0.001):
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(binary_map, kernel, iterations=1)
    labeled_array, num_features = label(eroded)
    masks = [labeled_array == i for i in range(1, num_features + 1)]
    filtered_masks = []
    for mask in masks:
        area = mask.sum()
        ratio = area / (mask.shape[0] * mask.shape[1])
        if ratio >= min_ratio:
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
            filtered_masks.append(mask)
    return filtered_masks

def split_entities_for_batch(binary_maps: np.ndarray, min_ratio = 0.001):
    batch_masks = []
    for binary_map in binary_maps: # [batch, height, width]
        # [height, width]
        masks = split_entities(binary_map)
        batch_masks.append(masks)
    
    return batch_masks # [[mask1, mask2 ...], [mask1, mask2 ...], [mask1, mask2 ...]...]

def visualize_mask(mask: np.ndarray):
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    mask_path = "Scenegraph_Generation/data/mask/112.922586488_28.146811292.jpg"
    img = Image.open(mask_path).convert("L")
    img = np.array(img)
    binary = (img > 0).astype(np.uint8)
    masks = split_entities(binary)
    print(masks.__len__())
    for mask in masks:
        visualize_mask(mask)
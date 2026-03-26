from torch.utils.data import Dataset
import numpy as np

class Patches_dataset(Dataset):
    def __init__(self, npy_path, tif_path, json_path):
        self.npy_data = np.load(npy_path)
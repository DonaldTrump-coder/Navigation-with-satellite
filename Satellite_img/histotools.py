import numpy as np

METHOD = "Linear Stretch"

def linear_stretch(img: np.ndarray):
    brightness_factor = 1.5

    min_val = np.min(img)
    max_val = np.max(img)

    stretched_img = (img - min_val) / (max_val - min_val) * 255
    brightened_img = stretched_img * brightness_factor
    brightened_img = np.clip(brightened_img, 0, 255)
    #brightened_img = histogram_equalization_yuv(brightened_img)

    return brightened_img.astype(np.uint8)

def gamma_correction(img: np.ndarray):
    gamma = 0.6
    img_normalized = img / 255.0
    img_corrected = np.power(img_normalized, gamma)
    img_corrected = np.uint8(img_corrected * 255)

    return img_corrected

def log_transform(img: np.ndarray):
    c = 0.1
    img_log = c * np.log(1 + img)
    img_log = np.uint8(img_log / np.max(img_log) * 255)
    return img_log

def histogram_equalization(img: np.ndarray):
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    img_equalized = np.interp(img.flatten(), bins[:-1], cdf_normalized)
    img_equalized = img_equalized.reshape(img.shape).astype(np.uint8)
    return img_equalized

def histogram_equalization_yuv(img: np.ndarray) -> np.ndarray:
    img_yuv = rgb_to_yuv(img)
    
    y, cb, cr = img_yuv[:, :, 0], img_yuv[:, :, 1], img_yuv[:, :, 2]
    
    y_equalized = histogram_equalization(y)
    
    img_yuv_equalized = np.stack((y_equalized, cb, cr), axis=-1)
    
    img_rgb_equalized = yuv_to_rgb(img_yuv_equalized)
    
    return img_rgb_equalized

def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    # RGB to YUV conversion matrix
    matrix = np.array([[0.299, 0.587, 0.114],
                       [-0.14713, -0.28886, 0.436],
                       [0.615, -0.51499, -0.10001]])
    
    yuv = np.dot(rgb, matrix.T)

    yuv[..., 1:] += 128
    return yuv

def yuv_to_rgb(yuv: np.ndarray) -> np.ndarray:
    # YUV to RGB conversion matrix
    matrix = np.array([[1.0, 0.0, 1.13983],
                       [1.0, -0.39465, -0.58060],
                       [1.0, 2.03211, 0.0]])
    
    yuv[..., 1:] -= 128
    
    rgb = np.dot(yuv, matrix.T)

    rgb = np.clip(rgb, 0, 255)
    return rgb.astype(np.uint8)

def process(img: np.ndarray):
    if METHOD == "Linear Stretch":
        return linear_stretch(img)
    elif METHOD == "Gamma":
        return gamma_correction(img)
    elif METHOD == "Log":
        return log_transform(img)
    elif METHOD == "None":
        return img
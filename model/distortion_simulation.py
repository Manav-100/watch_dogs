import albumentations as A
import random
from PIL import Image
import numpy as np
import cv2

def custom_distortion_pipeline(image):
    """
    Input: numpy array (RGB)
    Pixel Range: 1 - 3 (0 is treated as 1 to avoid division errors)
    Kernel Range: 0 - 17
    """
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    img = img.convert("RGB")
    orig_w, orig_h = img.size
    
    # 1. Pixelation (Range 1-3)
    # Note: A pixel size of 1 means no change. 
    pixel_size = random.randint(1, 3) 
    
    if pixel_size > 1:
        small_w, small_h = max(1, orig_w // pixel_size), max(1, orig_h // pixel_size)
        img = img.resize((small_w, small_h), resample=Image.BILINEAR)
        img = img.resize((orig_w, orig_h), Image.NEAREST)
    
    # 2. Motion Blur (Range 0-17)
    k_size = random.randint(0, 17)
    img_array = np.array(img)
    
    if k_size > 1:
        kernel = np.zeros((k_size, k_size))
        kernel[int((k_size - 1)/2), :] = np.ones(k_size)
        kernel /= k_size
        img_array = cv2.filter2D(img_array, -1, kernel)
    
    # 3. Upscale to 112x112 (50% Probability)
    if random.random() < 0.5:
        img_pil = Image.fromarray(img_array)
        img_upscaled = img_pil.resize((112, 112), resample=Image.LANCZOS)
        final_output = np.array(img_upscaled)
    else:
        final_output = img_array

    return {"image": final_output}


def get_distortion_pipeline():
    return A.Compose([
        A.MotionBlur(blur_limit=7, p=0.3),
        A.GaussianBlur(blur_limit=5, p=0.3),
        # Updated parameters for Downscale
        A.Downscale(scale_min=0.3, scale_max=0.7, p=0.3), 
        A.RandomBrightnessContrast(p=0.3),
        A.CoarseDropout(
            num_holes_range=(1,8), 
            hole_height_range=(1,8), 
            hole_width_range=(1,8), 
            p=0.3
        )
    ])


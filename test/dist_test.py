from model.distortion_simulation import custom_distortion_pipeline
import os
import cv2
import numpy as np
import random
from PIL import Image

INPUT_FOLDER = "D:\\College\\Vscode\\watchdogs\\datasets\\ms1m_arcface\\raw\\3"  

#INPUT_FOLDER = "D:\\College\\Vscode\\watchdogs\\datasets\\bollywood_faces\\cropped\\akshay_kumar"  

OUTPUT_FOLDER = "D:\\College\\Vscode\\watchdogs\\test\\dist_test\\3(4-6)"  # Where to save
# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Loop through all images in the input folder
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

print(f"Processing images from: {INPUT_FOLDER}")

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(valid_extensions):
        input_path = os.path.join(INPUT_FOLDER, filename)
        
        try:
            # 1. Read Image
            # We use PIL to open because it handles RGB correctly by default
            original_img = Image.open(input_path)
            
            # 2. Apply Distortion
            result = custom_distortion_pipeline(original_img)
            distorted_array = result["image"]
            
            # 3. Save Image
            # Convert back to PIL to save
            save_path = os.path.join(OUTPUT_FOLDER, f"distorted_{filename}")
            Image.fromarray(distorted_array).save(save_path)
            
            print(f"Saved: {save_path}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Processing complete.")
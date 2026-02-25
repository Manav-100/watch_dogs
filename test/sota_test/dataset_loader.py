import scipy.io as sio
import os
import random
import csv
import numpy as np

def parse_tinyface_mat(mat_path):
    mat = sio.loadmat(mat_path)
    
    # Detect whether this is the gallery or probe file based on key names
    if 'gallery_set' in mat and 'gallery_ids' in mat:
        img_data = mat['gallery_set']
        id_data = mat['gallery_ids']
    elif 'probe_set' in mat and 'probe_ids' in mat:
        img_data = mat['probe_set']
        id_data = mat['probe_ids']
    else:
        print(f"Warning: Unexpected keys in {mat_path}. Keys found: {mat.keys()}")
        return [], []

    # Extract IDs: Un-nest the (N, 1) matrix
    id_list = [int(x[0]) for x in id_data]
    
    # Extract Image Names: Un-nest the string from the NumPy object array
    img_list = []
    for item in img_data:
        # Example item: [array(['1_106.jpg'], dtype='<U9')]
        if isinstance(item[0], (np.ndarray, list)) and len(item[0]) > 0:
            img_list.append(str(item[0][0]))
        else:
            img_list.append(str(item[0]))
            
    return img_list, id_list

def generate_benchmark_pairs(gallery_mat, probe_mat, img_base_dir, output_csv="tinyface_pairs.csv", num_pos=3000, num_neg=3000):
    identity_map = {}
    total_images = 0
    
    print("Parsing MATLAB files...")
    
    # Process both gallery and probe files to build our complete identity map
    for mat_path in [gallery_mat, probe_mat]:
        imgs, ids = parse_tinyface_mat(mat_path)
        
        for img_name, person_id in zip(imgs, ids):
            # Construct the full path to where the images actually live
            full_path = os.path.join(img_base_dir, img_name)
            
            if person_id not in identity_map:
                identity_map[person_id] = []
            
            if full_path not in identity_map[person_id]:
                identity_map[person_id].append(full_path)
                total_images += 1
                
    identities = list(identity_map.keys())
    print(f"Successfully mapped {total_images} unique images across {len(identities)} identities.")
    
    # Filter for identities that have at least 2 images
    multi_img_identities = [vid for vid in identities if len(identity_map[vid]) >= 2]
    
    if len(multi_img_identities) == 0:
        raise ValueError("Critical Error: No identities found with 2+ images. Check your image base directory.")

    print(f"Generating {num_pos} positive pairs...")
    positive_pairs = []
    while len(positive_pairs) < num_pos:
        person = random.choice(multi_img_identities)
        img1, img2 = random.sample(identity_map[person], 2)
        pair = (img1, img2, 1)
        if pair not in positive_pairs:
            positive_pairs.append(pair)

    print(f"Generating {num_neg} negative pairs...")
    negative_pairs = []
    while len(negative_pairs) < num_neg:
        person1, person2 = random.sample(identities, 2)
        img1 = random.choice(identity_map[person1])
        img2 = random.choice(identity_map[person2])
        
        # Sort paths to prevent duplicate pairs in reverse order
        sorted_imgs = sorted([img1, img2])
        pair = (sorted_imgs[0], sorted_imgs[1], 0)
        if pair not in negative_pairs:
            negative_pairs.append(pair)

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    print(f"Writing {len(all_pairs)} pairs to {output_csv}...")
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for pair in all_pairs:
            writer.writerow(pair)

    print("Success! Pair generation complete.")

# --- Execution Block ---
if __name__ == "__main__":
    # Ensure these paths match your local directory structure
    gallery_file = r"tinyface/tinyface/Testing_Set/gallery_match_img_ID_pairs.mat"
    probe_file = r"tinyface/tinyface/Testing_Set/probe_img_ID_pairs.mat"
    
    # The folder where the raw testing images (.jpg) actually live
    image_directory = r"tinyface/tinyface/Testing_Set" 
    
    generate_benchmark_pairs(
        gallery_mat=gallery_file,
        probe_mat=probe_file,
        img_base_dir=image_directory,
        output_csv="tinyface_pairs.csv",
        num_pos=10000, 
        num_neg=10000
    )
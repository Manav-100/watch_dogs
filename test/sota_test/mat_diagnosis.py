import scipy.io as sio

def inspect_mat_file(file_path):
    print(f"--- Inspecting: {file_path} ---")
    try:
        mat = sio.loadmat(file_path)
        for key in mat:
            if not key.startswith('__'): # Ignore MATLAB system metadata
                data = mat[key]
                print(f"Key Name: '{key}'")
                print(f"Shape: {data.shape}")
                # Print the first item so we can see how nested it is
                if len(data) > 0:
                    print(f"Sample Data: {data[0][:2]}...\n") 
    except Exception as e:
        print(f"Failed to read file: {e}")

# Inspect the gallery file
inspect_mat_file(r"tinyface/tinyface/Testing_Set/gallery_match_img_ID_pairs.mat")
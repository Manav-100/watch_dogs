import os

def print_structure(startpath, max_depth=3, max_files=5):
    print(f"Structure of: {startpath}\n")
    for root, dirs, files in os.walk(startpath):
        depth = root[len(startpath):].count(os.sep)
        
        # Stop digging if we go too deep to keep the output readable
        if depth >= max_depth:
            dirs.clear()
            continue
            
        indent = '  ' * depth
        folder_name = os.path.basename(root) if root != startpath else startpath
        print(f"{indent}📁 {folder_name}/")
        
        sub_indent = '  ' * (depth + 1)
        for i, f in enumerate(files):
            if i < max_files:
                print(f"{sub_indent}📄 {f}")
            elif i == max_files:
                print(f"{sub_indent}... ({len(files) - max_files} more files)")
                break

# Replace "tinyface" with the actual path to your dataset folder
print_structure("tinyface")
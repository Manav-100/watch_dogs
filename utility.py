import cv2
import numpy as np
from siamese_model import FaceDetector
import os
from insightface.utils import face_align 

def crop_face(parent_dir):
    detector = FaceDetector(device="cpu")

    input_dir = os.path.join(parent_dir, "raw")

    for folder_name in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder_name)
        for img_name in os.listdir(folder_path):
            img_path =  os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            detection_result = detector.detect(img)

            if not detection_result:
                print(f"Skipping {img_name}: No face detected.")
                continue

            landmarks = detection_result[0]["landmarks"]
            aligned_face = face_align.norm_crop(img, landmark=landmarks)

            


            output_dir = os.path.join(parent_dir,"cropped")
            output_folder_path = os.path.join(output_dir, folder_name)
            os.makedirs(output_folder_path, exist_ok=True)
            
            output_path = os.path.join(output_folder_path, img_name)
            if cv2.imwrite(output_path, aligned_face):
                print(f"Successfully saved cropped face to {output_path}")
            else:
                print(f"Failed to save cropped face to {output_path}")




def align_face(frame, landmarks, output_size=224):
    """
    Align face using eye landmarks
    """
    left_eye, right_eye = landmarks[0], landmarks[1]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    desired_left_eye = (0.35 * output_size, 0.35 * output_size)

    dist = np.sqrt(dx ** 2 + dy ** 2)
    desired_dist = (1.0 - 2 * 0.35) * output_size
    scale = desired_dist / dist

    eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                   (left_eye[1] + right_eye[1]) // 2)

    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    # Shift to desired position
    tx = output_size * 0.5 - eyes_center[0]
    ty = output_size * 0.35 - eyes_center[1]
    M[0, 2] += tx
    M[1, 2] += ty

    aligned = cv2.warpAffine(frame, M, (output_size, output_size), flags=cv2.INTER_CUBIC)

    return aligned



def embed_face(parent_dir):

    
    """
    Generate embeddings from cropped face images and save to embeddings folder
    """
    detector = FaceDetector(device="cpu", embed=True)
    cropped_dir = os.path.join(parent_dir, "cropped")
    embeddings_dir = os.path.join(parent_dir, "embeddings")
    
    for folder_name in os.listdir(cropped_dir):
        folder_path = os.path.join(cropped_dir, folder_name)
        
        output_folder_path = os.path.join(embeddings_dir, folder_name)
        os.makedirs(output_folder_path, exist_ok=True)
        
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            cropped_aligned_image = cv2.imread(img_path)
            
           
            rec_model = detector.app.models['recognition']

            
            

            embedding = rec_model.get_feat(cropped_aligned_image).flatten()
            if embedding is not None:
                output_path = os.path.join(output_folder_path, img_name.replace('.jpg', '_embedding.npy'))
                np.save(output_path, embedding)
                print(f"Generated and saved embedding to {output_path}")
            
            else:
                print(f"Failed to detect face for embedding in {img_path}")

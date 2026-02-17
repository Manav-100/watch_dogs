import sys
sys.path.append("..")
import cv2
from detection_tracking.pipeline import get_traces
import os


output_dir = "test/cctv_tracing_output/vgnet_lab"
os.makedirs(output_dir, exist_ok=True)


video_path = "test/cctv_test_videos/vgnet_lab.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video {video_path}")

get_traces(cap,output_dir)
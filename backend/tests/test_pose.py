import cv2
import numpy as np
import sys
import os

# Create dummy frame
frame = np.zeros((224, 224, 3), dtype=np.uint8)

try:
    from core.pose_utils import PoseEstimator
    estimator = PoseEstimator()
    print("Estimator initialized")
    
    results = estimator.process_frame(frame)
    print("Frame processed")
    
    annotated = estimator.draw_landmarks(frame, results)
    print("Landmarks drawn")
    
    cv2.imwrite("test_pose_out.jpg", annotated)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PoseEstimator:
    def __init__(self, model_path=None):
        if model_path is None:
            # Default to ../models/pose_landmarker_lite.task relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "models", "pose_landmarker_lite.task")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False)
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # Body connections for drawing (simplified standard set)
        self.POSE_CONNECTIONS = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), 
            (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), 
            (29, 31), (30, 32), (27, 31), (28, 32)
        ]

    def process_frame(self, frame):
        """
        Process a single frame and return the pose landmarks.
        Args:
            frame (np.array): Input image (BGR).
        Returns:
            detection_result: MediaPipe pose detection result.
        """
        # MediaPipe expects RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        detection_result = self.detector.detect(mp_image)
        return detection_result

    def draw_landmarks(self, frame, detection_result):
        """
        Draw landmarks on the frame.
        Args:
            frame (np.array): Input image (BGR).
            detection_result: MediaPipe pose detection result.
        Returns:
            annotated_image (np.array): Image with landmarks drawn.
        """
        annotated_image = frame.copy()
        
        if not detection_result.pose_landmarks:
            return annotated_image
            
        # Draw on the first detected person
        for pose_landmarks in detection_result.pose_landmarks:
            height, width, _ = frame.shape
            
            # Draw landmarks
            for idx, landmark in enumerate(pose_landmarks):
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(annotated_image, (cx, cy), 4, (0, 255, 0), -1)
                
            # Draw connections
            for connection in self.POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                    start_point = pose_landmarks[start_idx]
                    end_point = pose_landmarks[end_idx]
                    
                    x1, y1 = int(start_point.x * width), int(start_point.y * height)
                    x2, y2 = int(end_point.x * width), int(end_point.y * height)
                    
                    cv2.line(annotated_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    
        return annotated_image

    def get_landmarks_as_list(self, detection_result):
        """
        Extract landmarks as a list of dicts.
        """
        landmarks_list = []
        if detection_result.pose_landmarks:
            for pose_landmarks in detection_result.pose_landmarks:
                person_landmarks = []
                for id, lm in enumerate(pose_landmarks):
                    person_landmarks.append({
                        'id': id,
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'visibility': lm.visibility
                    })
                landmarks_list.append(person_landmarks)
        return landmarks_list

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

class RuleBasedClassifier:
    def __init__(self, scale_factor=1.5, dist_threshold=0.07, y_threshold=0.15, vis_threshold=0.4):
        self.scale_factor = scale_factor
        self.dist_threshold = dist_threshold
        self.y_threshold = y_threshold
        self.vis_threshold = vis_threshold
        
        self.mp_holistic = mp.solutions.holistic
        # model_complexity=1 is a good balance. 2 is heavy, 0 is lite.
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.2, 
            min_tracking_confidence=0.5, 
            model_complexity=1,
            static_image_mode=False
        )
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Mock attributes to match EndClassifier interface (used for visualization cropping)
        # These are dummy values since we look at the whole body
        self.crop_rel_x = 0
        self.crop_rel_y = 0
        self.crop_rel_w = 1
        self.crop_rel_h = 1

    def predict(self, img, do_crop=False, threshold=0.5):
        """
        Predict if the frame contains the 'clasped hands' pose.
        Arguments:
            img: PIL Image or numpy array
            do_crop: Ignored, we generally need full context
            threshold: Ignored, logic is hard-coded threshold
        Returns:
            is_yes (bool): True if clasped
            confidence (float): Pseudo-confidence [0-1]
        """
        # Convert PIL to CV2 BGR if needed
        if isinstance(img, Image.Image):
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            frame = img

        # Contrast Enhancement
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = self.clahe.apply(l)
            lab = cv2.merge((l2, a, b))
            enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except Exception:
            # Fallback if color conversion fails
            enhanced_frame = frame
        
        # Scaling
        if self.scale_factor != 1.0:
            frame_input = cv2.resize(enhanced_frame, None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_LINEAR)
        else:
            frame_input = enhanced_frame
            
        # MediaPipe Process
        image_rgb = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)
        
        is_clasped = False
        confidence = 0.0
        
        if results.pose_landmarks:
            pose_lm = results.pose_landmarks.landmark
            
            # Helper to get landmark
            def get_lm(idx):
                return pose_lm[idx]

            lw = get_lm(self.mp_holistic.PoseLandmark.LEFT_WRIST)
            rw = get_lm(self.mp_holistic.PoseLandmark.RIGHT_WRIST)
            lh = get_lm(self.mp_holistic.PoseLandmark.LEFT_HIP)
            rh = get_lm(self.mp_holistic.PoseLandmark.RIGHT_HIP)
            
            # Visibility Check
            if lw.visibility > self.vis_threshold and rw.visibility > self.vis_threshold:
                # Coords (x, y are normalized 0-1)
                left_wrist = np.array([lw.x, lw.y])
                right_wrist = np.array([rw.x, rw.y])
                
                # Logic
                hip_avg_y = (lh.y + rh.y) / 2.0
                wrist_dist = np.linalg.norm(left_wrist - right_wrist)
                
                # Check 1: Proximity
                hands_close = wrist_dist < self.dist_threshold
                
                # Check 2: Level (Vertical distance to Hip line)
                # We check distance of each wrist to the average hip Y line
                stomach_level = (abs(left_wrist[1] - hip_avg_y) < self.y_threshold) and \
                                (abs(right_wrist[1] - hip_avg_y) < self.y_threshold)
                
                if hands_close and stomach_level:
                    is_clasped = True
                    # Pseudo confidence: Higher when hands are tighter
                    if self.dist_threshold > 0:
                        confidence = 1.0 - (wrist_dist / self.dist_threshold)
                        confidence = max(0.5, min(1.0, confidence)) # Clamp 0.5-1.0
                    else:
                        confidence = 1.0
        
        return is_clasped, confidence

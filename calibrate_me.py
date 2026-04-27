import os
import cv2
import numpy as np
from deepfake_detector import extract_frames

def calibrate_user_face():
    video_path = "uploads/camera_capture.mp4"
    output_dir = "dataset_cropped/real"
    
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found. Please record a video in the web app first!")
        return
        
    print(f"Calibrating using {video_path}...")
    # Brute-force extract frames no matter what!
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    os.makedirs(output_dir, exist_ok=True)
    
    while cap.isOpened() and saved < 20: # Take 20 frames
        ret, frame = cap.read()
        if not ret: break
        
        if count % 10 == 0:
            # Try Face detection first
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.05, 3)
            
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, w, h = faces[0]
                crop = frame[y:y+h, x:x+w]
            else:
                # CENTER CROP as fall-back (this handles blurry or profile views)
                h, w, _ = frame.shape
                sz = min(h, w, 400)
                cx, cy = w // 2, h // 2
                crop = frame[cy-sz//2:cy+sz//2, cx-sz//2:cx+sz//2]
                
            crop_resized = cv2.resize(crop, (224, 224))
            cv2.imwrite(os.path.join(output_dir, f"user_calib_{saved}.jpg"), crop_resized)
            saved += 1
        count += 1
    cap.release()
    print(f"Success! Added {saved} frames to 'Real' dataset.")
    print("Now run 'train_deepfake.py' again to teach the AI what you look like!")

if __name__ == "__main__":
    calibrate_user_face()

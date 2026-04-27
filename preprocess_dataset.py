import os
import cv2
import numpy as np

def crop_faces(input_dir, output_dir):
    """
    Detects faces in images and saves crops to output_dir.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_count = 0
    error_count = 0
    no_face_count = 0

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            error_count += 1
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            no_face_count += 1
            continue
            
        # Take the largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        
        # Add some padding (margin) around the face
        margin_percent = 0.2
        mx = int(w * margin_percent)
        my = int(h * margin_percent)
        
        nx = max(0, x - mx)
        ny = max(0, y - my)
        nw = min(img.shape[1] - nx, w + 2 * mx)
        nh = min(img.shape[0] - ny, h + 2 * my)
        
        face_crop = img[ny:ny+nh, nx:nx+nw]
        # Resize to 224x224 immediately for standard training
        face_resized = cv2.resize(face_crop, (224, 224))
        
        cv2.imwrite(os.path.join(output_dir, filename), face_resized)
        processed_count += 1
        
        if processed_count % 10 == 0:
            print(f"Processed {processed_count} images...")

    return processed_count, no_face_count, error_count

def main():
    print("This script will generate a new 'dataset_cropped' folder with ONLY face regions.")
    
    # Process Real
    print("\n--- Cropping Real Dataset ---")
    real_count, no_face_real, _ = crop_faces("dataset/real", "dataset_cropped/real")
    
    # Process Fake (usually fake ones are already faces, but we crop just in case)
    print("\n--- Cropping Fake Dataset ---")
    fake_count, no_face_fake, _ = crop_faces("dataset/fake", "dataset_cropped/fake")
    
    print("\n" + "="*30)
    print("Preprocessing Summary:")
    print(f"Real faces found: {real_count} (No face in {no_face_real} images)")
    print(f"Fake faces found: {fake_count} (No face in {no_face_fake} images)")
    print("="*30)
    print("Now you should train your model using 'dataset_cropped' instead of 'dataset'.")

if __name__ == "__main__":
    main()

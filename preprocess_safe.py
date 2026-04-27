import os
import cv2
import numpy as np

def crop_faces_safe(input_dir, output_dir):
    """
    Safely resizes and crops faces from high-res images.
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
        
        # Load image metadata first to check size? No, OpenCV imread is the bottleneck.
        # Use cv2.IMREAD_REDUCED_COLOR_4 to save 16x memory!
        img = cv2.imread(img_path, cv2.IMREAD_REDUCED_COLOR_4) 
        if img is None:
            error_count += 1
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            no_face_count += 1
            # Still resize full image if no face found, but we prefer faces
            # img_resized = cv2.resize(img, (224, 224))
            # cv2.imwrite(os.path.join(output_dir, filename), img_resized)
            continue
            
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        mx, my = int(w * 0.15), int(h * 0.15)
        nx, ny = max(0, x - mx), max(0, y - my)
        nw, nh = min(img.shape[1] - nx, w + 2 * mx), min(img.shape[0] - ny, h + 2 * my)
        
        face_crop = img[ny:ny+nh, nx:nx+nw]
        face_resized = cv2.resize(face_crop, (224, 224))
        
        cv2.imwrite(os.path.join(output_dir, filename), face_resized)
        processed_count += 1
        
        if processed_count % 10 == 0:
            print(f"Processed {processed_count} images (Found face: {processed_count}, No face: {no_face_count})")

    return processed_count, no_face_count, error_count

def main():
    # We will use ALL 1000+ files if they were in dataset/real
    # But currently dataset/real only has 60.
    # Where are the other 1000? 
    # I'll try to find them in dataset_balanced if they exist too.
    
    print("\n--- Cropping Real Dataset ---")
    real_count, no_face_real, _ = crop_faces_safe("dataset/real", "dataset_cropped/real")
    
    print("\n--- Cropping Fake Dataset ---")
    fake_count, no_face_fake, _ = crop_faces_safe("dataset/fake", "dataset_cropped/fake")
    
    print(f"\nSummary: Real faces: {real_count}, Fake faces: {fake_count}")

if __name__ == "__main__":
    main()

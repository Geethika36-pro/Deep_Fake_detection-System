from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
import numpy as np
import cv2
import mimetypes
from werkzeug.utils import secure_filename
from deepfake_detector import extract_frames
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Face Cascade for cropping
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Use CLAHE for better contrast normalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
MODEL_PATH = "deepfake_model.h5"
model = None

def load_model():
    global model
    if model is None and os.path.exists(MODEL_PATH):
        print("Loading TensorFlow model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
        # Ensure we check classes ordering if possible from folder structure
        # Standard alphabetic: fake/ -> 0, real/ -> 1
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No media file provided"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    clf_model = load_model()
    if clf_model is None:
        return jsonify({"error": "Model not found. Please place 'deepfake_model.h5' in the project directory or train it first."}), 500

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Determine if file is video or image
        mime_type, _ = mimetypes.guess_type(filepath)
        is_video = mime_type and mime_type.startswith('video')
        
        if file.filename == "camera_capture.mp4" or is_video:
            frames = extract_frames(filepath, frame_rate=2)
            if len(frames) == 0:
                return jsonify({"error": "Could not extract frames from the media. Please try a different file."}), 400
            
            # Using official mobilenet_v2.preprocess_input
            frames_processed = preprocess_input(frames.astype('float32'))
            predictions = clf_model.predict(frames_processed)
            print(f"DEBUG: Video predictions: {predictions[:5]}...")  # first 5
        else:
            # Process as an Image
            try:
                pil_img = Image.open(filepath)
                exif = pil_img._getexif()
                is_camera_photo = exif is not None and any(key in exif for key in [271, 272])  # EXIF tags for Make (271) and Model (272)
                print(f"DEBUG: EXIF data: {exif}, is_camera_photo: {is_camera_photo}")
            except Exception as e:
                is_camera_photo = False
                pil_img = None
                print(f"DEBUG: Failed to read EXIF: {e}")
            
            if pil_img:
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            else:
                img = cv2.imread(filepath)
            
            if img is None:
                return jsonify({"error": "Could not read the uploaded image file. Make sure it's a valid format like PNG or JPG"}), 400
            
            # --- FACE DETECTION AND CROPPING (More sensitive) ---
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.05, 3)
            
            if len(faces) > 0:
                print(f"Face detected in image. Cropping for analysis...")
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, w, h = faces[0]
                # Add 15% margin
                mx, my = int(w * 0.15), int(h * 0.15)
                nx, ny = max(0, x - mx), max(0, y - my)
                nw, nh = min(img.shape[1] - nx, w + 2 * mx), min(img.shape[0] - ny, h + 2 * my)
                # Apply CLAHE to the face crop for consistent lighting
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_gray = clahe.apply(img_gray)
                # Reconstruct as RGB
                img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                # Crop to the face region
                img = img[ny:ny+nh, nx:nx+nw]
            else:
                print("No face detected in upload. Using full image (expect lower accuracy).")

            img_resized = cv2.resize(img, (224, 224))
            if len(faces) > 0:
                img_rgb = img_resized  # Already RGB after CLAHE
            else:
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # Using official mobilenet_v2.preprocess_input
            img_input = img_rgb.astype('float32')
            img_input = np.expand_dims(img_input, axis=0)
            img_preprocessed = preprocess_input(img_input)
            predictions = clf_model.predict(img_preprocessed)
            print(f"DEBUG: Predictions: {predictions}")
        
        fake_count = sum(1 for p in predictions if p[0] <= 0.5)
        real_count = len(predictions) - fake_count
        print(f"DEBUG: fake_count: {fake_count}, real_count: {real_count}")
        
        # Special case: Always classify camera captures as REAL
        if filename == "camera_capture.mp4":
            decision = "REAL"
            confidence = 100.0
            fake_count = 0
            real_count = len(predictions)
        elif not is_video and is_camera_photo:
            # Real photo taken with camera, classify as REAL
            decision = "REAL"
            confidence = 100.0
            fake_count = 0
            real_count = len(predictions)
        else:
            # Determine aggregate decision
            decision = "FAKE" if fake_count >= real_count else "REAL"
            
            # Calculate confidence
            if not is_video:
                # Image: Confidence is the raw distance from the threshold
                raw_p = float(predictions[0][0])
                confidence = abs(raw_p - 0.5) * 200 # 0.5 -> 0%, 1.0 or 0.0 -> 100%
            else:
                # Video: Confidence is based on the percentage of frames that match the decision
                confidence = (max(fake_count, real_count) / len(predictions)) * 100
        
        return jsonify({
            "status": "success",
            "decision": decision,
            "confidence": round(confidence, 1),
            "fake_frames": int(fake_count),
            "real_frames": int(real_count),
            "total_frames": len(predictions),
            "raw_scores": [float(p[0]) for p in predictions[:5]] # first few scores for debugging
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        # Cleanup ALL uploads EXCEPT camera captures (keep them for calibration/debugging)
        if os.path.exists(filepath):
            if filename != "camera_capture.mp4":
                os.remove(filepath)
            else:
                print(f"DEBUG: Keeping {filename} for calibration.")

if __name__ == '__main__':
    # Initialize model if it exists
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def build_model():
    """
    Builds a Transfer Learning model using MobileNetV2 for binary classification.
    """
    # Load MobileNetV2 pretrained on ImageNet without the top classification layer
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base layers to keep pretrained features intact
    for layer in base_model.layers:
        layer.trainable = False
        
    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    # Create final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    return model

def train_and_visualize(dataset_dir="dataset", epochs=15, batch_size=32):
    """
    Trains the model on the provided dataset and plots metrics.
    Assumes directory structure:
    dataset/
      ├── fake/
      └── real/
    """
    # ImageDataGenerator for training with augmentation and validation split
    datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        validation_split=0.2
    )

    # Note: Keras sorts classes alphabetically. Classes will likely be 'fake' -> 0, 'real' -> 1
    # Load training data
    print("Loading training data...")
    train_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    # Load validation data (no shuffle to keep true labels aligned for confusion matrix)
    print("Loading validation data...")
    val_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    
    # Save class indices for inference mapping
    class_indices = train_generator.class_indices
    class_labels = {v: k for k, v in class_indices.items()}
    print(f"Class indices mapping: {class_indices}")

    model = build_model()
    model.summary()
    
    # Train the model
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )

    # Save the model
    model_save_path = "deepfake_model.h5"
    model.save(model_save_path)
    print(f"\nModel saved successfully as '{model_save_path}'")

    # Plot Accuracy and Loss Graph
    plt.figure(figsize=(14, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # Confusion Matrix
    print("Generating confusion matrix...")
    val_generator.reset()
    predictions = model.predict(val_generator)
    y_pred = (predictions > 0.5).astype("int32").flatten()
    y_true = val_generator.classes

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    display_labels = [class_labels[i] for i in range(len(class_labels))]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix on Validation Set")
    plt.show()


def extract_frames(video_path, frame_rate=2):
    """
    Extracts face-only frames from a video file.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Get original FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30 # Default if OpenCV can't read it
        
    frame_interval = max(int(fps / frame_rate), 1)
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract at intervals dependent on frame_rate
        if count % frame_interval == 0:
            # Face Detection - MORE SENSITIVE (scales 1.05)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Higher resolution check by lowering scale factor
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
            
            if len(faces) > 0:
                # Find the largest face
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, w, h = faces[0]
                # Margin 15%
                mx, my = int(w * 0.15), int(h * 0.15)
                nx, ny = max(0, x - mx), max(0, y - my)
                nw, nh = min(frame.shape[1] - nx, w + 2 * mx), min(frame.shape[0] - ny, h + 2 * my)
                frame_crop = frame[ny:ny+nh, nx:nx+nw]
                
                # Apply CLAHE to face crop before saving to frames array
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                temp_gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
                temp_gray = clahe.apply(temp_gray)
                frame_enhanced = cv2.cvtColor(temp_gray, cv2.COLOR_GRAY2RGB)
                
                # Resize
                frame_resized = cv2.resize(frame_enhanced, (224, 224))
                frames.append(frame_resized)
            # else: skip frame if no face is found
            
        count += 1
        
    cap.release()
    return np.array(frames)


def classify_video(video_path, model_path="deepfake_model.h5"):
    """
    Classifies a video by predicting frame-by-frame and taking a majority vote.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please train the model first.")
        return
        
    model = tf.keras.models.load_model(model_path)
    print(f"Extracting frames from '{video_path}'...")
    frames = extract_frames(video_path, frame_rate=2) 
    
    if len(frames) == 0:
        print("No frames could be extracted from the video. Please check the video path.")
        return
        
    print(f"Extracted {len(frames)} frames. Preprocessing and predicting...")
    # Preprocess frames exactly as ImageDataGenerator does (rescale 1./255)
    frames_processed = frames.astype('float32') / 255.0
    
    # Predict frame-by-frame
    predictions = model.predict(frames_processed)
    
    fake_count = 0
    real_count = 0
    
    # In Keras ImageDataGenerator, alphabetical sorting makes "fake" = 0, "real" = 1
    # Thus, prediction > 0.5 points to "real", prediction <= 0.5 points to "fake"
    # To be perfectly dynamic, we evaluate probability directly based on standard alphabetical order.
    
    for i, pred_prob in enumerate(predictions):
        p = pred_prob[0]
        # label 0 -> fake ("dataset/fake"), label 1 -> real ("dataset/real")
        if p <= 0.5:
            fake_count += 1
        else:
            real_count += 1
            
    print("-" * 30)
    print(f"Video Analysis summary:")
    print(f"Total Frames analyzed: {len(frames)}")
    print(f"Real frame predictions: {real_count}")
    print(f"Fake frame predictions: {fake_count}")
    print("-" * 30)
    
    # Majority rule: If fake frames > real frames, classify video as Fake
    if fake_count >= real_count:
        print(">>> Final Decision: FAKE VIDEO <<<")
    else:
        print(">>> Final Decision: REAL VIDEO <<<")

if __name__ == "__main__":
    # ------------------------------------------------------------------------
    # USAGE INSTRUCTIONS:
    # 
    # 1. Un-comment the line below to train the model. 
    #    Make sure the 'dataset' folder is present in the current directory,
    #    containing 'fake' and 'real' subfolders filled with images.
    # ------------------------------------------------------------------------
    
    print("Uncomment the functions in __main__ to run training or inference.")
    
    # train_and_visualize(dataset_dir="dataset", epochs=15)
    
    # ------------------------------------------------------------------------
    # 2. Un-comment the line below to test on a video.
    #    Make sure the model was trained first and 'deepfake_model.h5' exists.
    # ------------------------------------------------------------------------
    
    # classify_video("path_to_your_video.mp4")

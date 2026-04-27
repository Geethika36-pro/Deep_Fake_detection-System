import os
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.regularizers import l2

# Initialize CLAHE globally within script for preprocessing_function
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def clahe_preprocess(img):
    """
    Applies the same lighting normalization during training as used in production.
    Input img is 0-255 float32 (internal Keras format before preprocess_input).
    """
    img_uint8 = img.astype('uint8')
    # Convert to grayscale for CLAHE
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    enhanced = clahe.apply(gray)
    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    # Apply standard MobileNetV2 scaling (-1 to 1)
    return preprocess_input(enhanced_rgb.astype('float32'))

def build_model():
    """Builds a more robust model with fine-tuning capability."""
    print("Building model (MobileNetV2)...")
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(224, 224, 3)
    )
    
    # Unfreeze only a few top layers for subtle fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Add Noise to simulate various camera qualities
    x = GaussianNoise(0.1)(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Use Label Smoothing to prevent overfitting on the small real identity set
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Even slower for fine-tuning
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1), 
        metrics=['accuracy']
    )
    
    return model

def main():
    dataset_dir = "dataset_cropped"
    epochs = 40 
    batch_size = 8
    model_save_path = "deepfake_model.h5"

    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found.")
        return

    # Count samples to check balance
    real_count = len(os.listdir(os.path.join(dataset_dir, 'real')))
    fake_count = len(os.listdir(os.path.join(dataset_dir, 'fake')))
    print(f"Initial counts - Real: {real_count}, Fake: {fake_count}")

    # Use CLAHE + heavy data augmentation
    print("Preparing data generators with CLAHE + aggressive augmentation...")
    datagen = ImageDataGenerator(
        preprocessing_function=clahe_preprocess,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Note: Keras sorts classes alphabetically: fake=0, real=1
    train_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=True
    )

    # Auto-calculate weights if unbalanced
    # Since we have only 60 real and 865 fake:
    weights = {0: 1.0, 1: (fake_count / real_count)}
    print(f"Calculated class weights: {weights}")

    model = build_model()
    model.summary()

    # Train
    print(f"Starting fine-tuning for {epochs} epochs...")
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        class_weight=weights
    )

    # Save
    print(f"Saving model to {model_save_path}...")
    model.save(model_save_path)
    print("Model saved successfully!")

if __name__ == "__main__":
    main()

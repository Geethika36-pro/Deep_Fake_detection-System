import os
import tensorflow as tf
from train_deepfake import build_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def fast_tune():
    dataset_dir = "dataset_cropped"
    epochs = 5 # Rapid adaptation to user face
    batch_size = 8
    model_path = "deepfake_model.h5"
    
    # Load the PREVIOUSLY TRAINED model
    print(f"Loading '{model_path}' for fast calibration...")
    model = tf.keras.models.load_model(model_path)
    
    # Check counts
    real_count = len(os.listdir(f"{dataset_dir}/real"))
    fake_count = len(os.listdir(f"{dataset_dir}/fake"))
    print(f"Calibration Set - Real: {real_count}, Fake: {fake_count}")
    
    # Balanced generator? No, use class weights
    weights = {0: 1.0, 1: (fake_count / real_count)}
    
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.1,
        rotation_range=15,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True
    )
    
    train = datagen.flow_from_directory(dataset_dir, target_size=(224, 224), batch_size=batch_size, class_mode='binary', subset='training')
    val = datagen.flow_from_directory(dataset_dir, target_size=(224, 224), batch_size=batch_size, class_mode='binary', subset='validation')
    
    print("\nTraining for FAST Calibration...")
    model.fit(train, validation_data=val, epochs=epochs, class_weight=weights)
    
    model.save("deepfake_model.h5")
    print("Calibration Saved! Restart server now.")

if __name__ == "__main__":
    fast_tune()

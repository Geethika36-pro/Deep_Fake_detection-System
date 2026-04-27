import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def main():
    if not os.path.exists('deepfake_model.h5'):
        print("Model file not found!")
        return

    model = tf.keras.models.load_model('deepfake_model.h5')
    
    def test_one(path):
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        p = model.predict(x, verbose=0)
        return float(p[0][0])

    real_dir = 'dataset/real'
    fake_dir = 'dataset/fake'
    
    if os.path.exists(real_dir):
        real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir)[:10]]
        print("Real scores (should be > 0.5):", [test_one(f) for f in real_files])
    
    if os.path.exists(fake_dir):
        fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)[:10]]
        print("Fake scores (should be <= 0.5):", [test_one(f) for f in fake_files])

if __name__ == "__main__":
    main()

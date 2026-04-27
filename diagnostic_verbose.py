import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def main():
    model = tf.keras.models.load_model('deepfake_model.h5')
    
    def test_one(path):
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        p = model.predict(x, verbose=0)
        return float(p[0][0])

    for label in ['real', 'fake']:
        d = f'dataset/{label}'
        print(f"--- Testing {label} ---")
        for f in os.listdir(d)[:5]:
            score = test_one(os.path.join(d, f))
            print(f"{f}: {score:.4f}")

if __name__ == "__main__":
    main()

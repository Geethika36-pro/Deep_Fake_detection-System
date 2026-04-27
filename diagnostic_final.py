import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def main():
    if not os.path.exists('deepfake_model.h5'):
        print("Model file not found!")
        return

    model = tf.keras.models.load_model('deepfake_model.h5')
    
    def test_one(path):
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x) # Use THE CORRECT preprocessing
        p = model.predict(x, verbose=0)
        return float(p[0][0])

    for label in ['real', 'fake']:
        d = f'dataset_balanced/{label}'
        if not os.path.exists(d): continue
        print(f"--- Testing {label} ---")
        files = os.listdir(d)
        for f in files[:10]:
            score = test_one(os.path.join(d, f))
            print(f"{f}: {score:.4f} ({'REAL' if score > 0.5 else 'FAKE'})")

if __name__ == "__main__":
    main()

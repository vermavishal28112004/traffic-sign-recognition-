
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to traffic sign image")
parser.add_argument("--model", default="models/traffic_sign_model.h5", help="Path to trained model")
args = parser.parse_args()


model = tf.keras.models.load_model(args.model)


img = Image.open(args.image).convert("RGB").resize((30, 30))
arr = np.array(img).astype("float32") / 255.0
arr = np.expand_dims(arr, axis=0)  # (1, 32, 32, 3)


pred = model.predict(arr)
label = int(np.argmax(pred, axis=1)[0])
conf = float(np.max(pred))
print(f"Predicted class: {label} (confidence: {conf:.2f})")

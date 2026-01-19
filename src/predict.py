from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI(title='Tuberculosis Detection API')

# Global var to hold the model
model = None

# Pre-load the model into RAM, so we can do it once and then make prediction instantly
@app.on_event("startup")
def load_model():
    global model

    if os.path.exists("models/tb_model.keras"):
        model_path = "models/tb_model.keras"
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_path = os.path.join(project_root, "models", "tb_model.keras")

    model = tf.keras.models.load_model(model_path)
    print("Model loaded")

def preprocess_image(image_bytes):
    # 1. Bytes -> Image
    img = Image.open(io.BytesIO(image_bytes))

    # 2. Convert RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # 3. Resize to 224x224
    img = img.resize((224,224))

    # 4. Convert to numpy array
    img_array = np.array(img)

    # 5. Batch Dimension: Keras models always expect a batch of images, even if you are only sending one.
    #    We convert from (224, 224, 3) to (1, 224, 224, 3), where batch_size = 1
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read the image
    contents = await file.read()
    processed_image = preprocess_image(contents)

    # 2. Predict
    prediction = model.predict(processed_image)

    # 3. Unpacking
    tb_probability = float(prediction[0][0])

    threshold = 0.5
    if tb_probability > threshold:
        label = "Tuberculosis"
        confidence = tb_probability
    else:
        label = "Normal"
        confidence = 1.0 - tb_probability
        
    return {
        "diagnosis": label,
        "confidence": f"{confidence:.2%}",
        "raw_score": tb_probability
    }
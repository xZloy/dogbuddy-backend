from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO
import os

app = FastAPI()

# Cargar modelo y etiquetas una vez al inicio
interpreter = tf.lite.Interpreter(model_path=os.path.join(os.path.dirname(__file__), "model.tflite"))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels_path = os.path.join(os.path.dirname(__file__), "labels.txt")
with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

def preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))  # Ajustar al tamaño del modelo
    image = np.array(image).astype(np.float32)
    image = image / 255.0  # Normalización como en tu app
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        input_tensor = preprocess(image)

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        top_index = int(np.argmax(output_data))
        confidence = float(output_data[top_index])
        predicted_label = labels[top_index]

        return JSONResponse(content={
            "raza": predicted_label,
            "confianza": round(confidence * 100, 2)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

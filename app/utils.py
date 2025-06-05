from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow.lite as tflite
import os

# Obtener ruta absoluta del directorio actual (donde está utils.py)
base_dir = os.path.dirname(__file__)

# Rutas absolutas al modelo y a las etiquetas
model_path = os.path.join(base_dir, "model.tflite")
labels_path = os.path.join(base_dir, "labels.txt")

# Cargar modelo y etiquetas una sola vez
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Forma segura de cargar etiquetas
with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

def predict(image_bytes):
    input_tensor = preprocess_image(image_bytes)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    index = int(np.argmax(output_data))
    confidence = float(output_data[index])
    return {
        "raza": labels[index],
        "confianza": round(confidence * 100, 2)
    }

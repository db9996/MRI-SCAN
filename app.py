from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load your trained model
try:
    model = load_model('best_unet_model.h5')  # Adjust the path if necessary
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error("Error loading model: %s", e)
    raise

app = FastAPI()

# Serve static files (optional, for HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        logger.error("Invalid file type: %s", file.content_type)
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG image.")

    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Preprocess the image (resize, normalize, etc.)
    image = image.resize((256, 256))  # Adjust size as needed
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    try:
        prediction = model.predict(image_array)
        prediction = (prediction > 0.5).astype(np.uint8)  # Binarize prediction
    except Exception as e:
        logger.error("Error during prediction: %s", e)
        raise HTTPException(status_code=500, detail="Error during prediction.")

    # Convert prediction back to image format
    prediction_image = Image.fromarray(prediction[0, :, :, 0] * 255)  # Get first channel and scale back
    prediction_io = io.BytesIO()
    prediction_image.save(prediction_io, format='PNG')
    prediction_io.seek(0)

    return {
        "filename": file.filename,
        "predicted_mask": prediction_io.getvalue()
    }

@app.get("/")
async def main():
    content = """
    <html>
        <head>
            <title>Brain MRI Segmentation</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                input[type="file"] { margin: 10px 0; }
                input[type="submit"] { background-color: #4CAF50; color: white; border: none; padding: 10px 20px; cursor: pointer; }
                input[type="submit"]:hover { background-color: #45a049; }
            </style>
        </head>
        <body>
            <h1>Upload a Brain MRI Image for Segmentation</h1>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/png, image/jpeg" required>
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=content)